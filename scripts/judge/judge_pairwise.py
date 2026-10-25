#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent pairwise judge for DPO / best-of-N.

Why: the project's RM proxy and "gold" RM are both trained on UltraFeedback with
the same model family and the same length bias. They cannot serve as an
independent judge. This script calls either an API model (OpenAI-compatible) or
a local HF model, asks it to compare A/B, swaps positions to cancel position
bias, and reports win rate + length bins.

Input JSONL (one record per line):
  {"id": "...", "prompt": "...", "response_a": "...", "response_b": "...",
   "label_a": "base", "label_b": "dpo"}

Usage (API, recommended):
  export JUDGE_API_KEY=sk-...
  export JUDGE_BASE_URL=https://api.deepseek.com/v1   # or OpenAI
  export JUDGE_MODEL=deepseek-chat
  python scripts/judge/judge_pairwise.py \
    --pairs /root/autodl-tmp/work/results/dpo_pairs.jsonl \
    --out /root/autodl-tmp/work/results/dpo_judge.json \
    --backend api --votes 3

Usage (local 7B, 24G):
  python scripts/judge/judge_pairwise.py \
    --pairs ... --out ... --backend local \
    --model Qwen/Qwen2.5-7B-Instruct --load-4bit

Output JSON:
  {"summary": {...}, "pairs": [{"id":..., "final_winner":"B", "votes":[...]}]}
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any

JUDGE_SYSTEM = (
    "You are a strict, independent evaluator for LLM responses. "
    "Compare Response A and Response B for the same user request. "
    "Judge factual correctness, instruction following, helpfulness, and safety. "
    "Ignore length, verbosity, confidence, formatting, and model identity. "
    "If the two are genuinely equivalent, answer tie. "
    'Return JSON only: {"winner": "A" | "B" | "tie", "reason": "<=30 words"}'
)

JUDGE_USER = """User request:
{prompt}

Response A:
{a}

Response B:
{b}

Which response is better? Return JSON only."""


def _norm_winner(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ("a", "response a", "left"):
        return "A"
    if x in ("b", "response b", "right"):
        return "B"
    if x in ("tie", "equal", "draw", "same"):
        return "tie"
    m = re.search(r'"winner"\s*:\s*"?(a|b|tie)', x, re.I)
    if m:
        return m.group(1).upper() if m.group(1).lower() != "tie" else "tie"
    m = re.search(r"\b(a|b|tie)\b", x, re.I)
    return (m.group(1).upper() if m and m.group(1).lower() != "tie" else "tie") if m else "tie"


def _word_len(text: str) -> int:
    return len((text or "").split())


class APIPairJudge:
    def __init__(self, model: str, base_url: str | None, api_key: str | None,
                 temperature: float = 0.2, max_tokens: int = 256):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("pip install openai  # required for --backend api") from e
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def judge(self, prompt: str, a: str, b: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(prompt=prompt, a=a, b=b)},
        ]
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                return _norm_winner(text), text.strip()[:500]
            except Exception as e:  # noqa: BLE001
                err = str(e)
                low = err.lower()
                # DeepSeek: 402 Insufficient Balance — retrying will not help.
                if "402" in low or "insufficient balance" in low:
                    raise SystemExit(
                        "Judge API call failed: 402 Insufficient Balance.\n"
                        "Recharge at https://platform.deepseek.com (¥10 is enough), "
                        "or run with --backend local.\n"
                        f"Raw error: {err[:400]}"
                    )
                if any(x in low for x in ["401", "403", "authentication",
                                          "invalid api key", "unauthorized"]):
                    raise SystemExit(
                        "Judge API auth failed (401/403).\n"
                        f"Raw error: {err[:400]}\n"
                        "Check JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL.\n"
                        "DeepSeek: base=https://api.deepseek.com/v1 model=deepseek-chat\n"
                        "OpenAI:   base=https://api.openai.com/v1   model=gpt-4o-mini"
                    )
                if "422" in low:
                    raise SystemExit(
                        "Judge API 422 (bad request). If messages was a string, "
                        "the client is misconfigured.\n"
                        f"Raw error: {err[:400]}"
                    )
                if attempt == 2:
                    return "error", f"ERROR: {err}"
                time.sleep(2 ** attempt)
        return "error", "ERROR"


class LocalPairJudge:
    def __init__(self, model: str, load_4bit: bool = False,
                 temperature: float = 0.2, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def judge(self, prompt: str, a: str, b: str) -> tuple[str, str]:
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(prompt=prompt, a=a, b=b)},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                top_p=0.9,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        gen = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return _norm_winner(gen), gen.strip()[:500]


def make_judge(args):
    if args.backend == "api":
        key = args.api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set JUDGE_API_KEY (or OPENAI_API_KEY) for --backend api")
        base_url = args.base_url or os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
        model = args.model or os.environ.get("JUDGE_MODEL", "deepseek-chat")
        return APIPairJudge(model, base_url, key, args.temperature, args.max_tokens)
    return LocalPairJudge(args.model or "Qwen/Qwen2.5-7B-Instruct",
                          args.load_4bit, args.temperature, args.max_tokens)


def judge_one(judge, prompt: str, a: str, b: str, votes: int, seed: int) -> dict:
    """Run position-swapped votes; return final winner in original A/B space."""
    rng = random.Random(seed)
    raw = []
    for v in range(votes):
        swap = (v % 2 == 1) if votes > 1 else False
        # For votes > 2, also randomize a bit to avoid always same order
        if votes > 2 and v >= 2:
            swap = rng.random() < 0.5
        if swap:
            w, reason = judge.judge(prompt, b, a)
            w = {"A": "B", "B": "A", "tie": "tie", "error": "error"}[w]
        else:
            w, reason = judge.judge(prompt, a, b)
        if w not in ("A", "B", "tie", "error"):
            w = "error"
        raw.append({"order": "BA" if swap else "AB", "winner_original": w,
                    "reason": reason})
    cnt = Counter(x["winner_original"] for x in raw)
    if cnt.get("error", 0) == len(raw):
        final = "error"
    else:
        top = [(k, v) for k, v in cnt.most_common() if k != "error"]
        if top and (len(top) == 1 or top[0][1] > (top[1][1] if len(top) > 1 else 0)):
            final = top[0][0]
        else:
            final = "tie"
    return {"votes": raw, "final_winner": final, "vote_counts": dict(cnt)}


def summarize(records: list[dict], label_a: str, label_b: str) -> dict:
    n = len(records)
    errors = sum(r["final_winner"] == "error" for r in records)
    valid = n - errors
    wins_b = sum(r["final_winner"] == "B" for r in records)
    wins_a = sum(r["final_winner"] == "A" for r in records)
    ties = valid - wins_a - wins_b
    bins = {"a_longer": [0, 0], "similar": [0, 0], "b_longer": [0, 0]}  # [n, wins_b]
    lens_a, lens_b = [], []
    for r in records:
        la, lb = _word_len(r["response_a"]), _word_len(r["response_b"])
        lens_a.append(la)
        lens_b.append(lb)
        diff = lb - la
        key = "b_longer" if diff > 20 else ("a_longer" if diff < -20 else "similar")
        bins[key][0] += 1
        bins[key][1] += int(r["final_winner"] == "B")
    def rate(x):
        return round(x, 4) if x == x else None
    return {
        "n": n,
        "valid": valid,
        "errors": errors,
        "label_a": label_a,
        "label_b": label_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        f"win_rate_{label_b}_excl_ties": rate(wins_b / max(wins_b + wins_a, 1)),
        f"win_rate_{label_b}_incl_ties": rate((wins_b + 0.5 * ties) / max(valid, 1)),
        "avg_words_a": rate(sum(lens_a) / max(n, 1)),
        "avg_words_b": rate(sum(lens_b) / max(n, 1)),
        "avg_word_diff_b_minus_a": rate((sum(lens_b) - sum(lens_a)) / max(n, 1)),
        "length_bins": {
            k: {"n": v[0], "wins_b": v[1],
                "win_rate_b": rate(v[1] / max(v[0], 1))} for k, v in bins.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSONL with prompt/response_a/response_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["api", "local"], default="api")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--votes", type=int, default=2, help="2 = both orders; 3 = + one extra")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    pairs = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[data] pairs={len(pairs)} backend={args.backend}", flush=True)

    judge = make_judge(args)
    records = []
    for i, p in enumerate(pairs):
        rec = judge_one(judge, p["prompt"], p["response_a"], p["response_b"],
                        args.votes, args.seed + i)
        records.append({**p, **rec})
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"[judge] {i+1}/{len(pairs)}", flush=True)

    label_a = pairs[0].get("label_a", "A") if pairs else "A"
    label_b = pairs[0].get("label_b", "B") if pairs else "B"
    summary = summarize(records, label_a, label_b)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": records}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"[done] -> {args.out}", flush=True)
    if summary["errors"] == summary["n"]:
        raise SystemExit("ALL judge votes failed; do not use this file. See reasons above.")


if __name__ == "__main__":
    sys.exit(main())
