#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI-compatible pairwise judge using only urllib (no openai package).

Supports:
  - any /chat/completions endpoint (Zhipu, DeepSeek, OpenAI, vLLM server)
  - parallel workers (API latency bound)
  - prompt styles: default | correctness_first
  - extra body fields, e.g. {"thinking": {"type": "disabled"}} for Zhipu GLM

Output schema matches judge_pairwise.py ({"summary":..., "pairs":[...]}),
so aggregate_judges.py can read both.

Usage:
  python scripts/judge/judge_api.py --pairs p2/judge_v2/raw/dpo800_pairs.jsonl \
    --out p2/judge_v2/dpo800_zhipu.json --base-url https://open.bigmodel.cn/api/paas/v4 \
    --model glm-4.5-air --api-key-env ZHIPU_API_KEY --votes 3 --workers 12 \
    --extra-json '{"thinking":{"type":"disabled"}}'
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_pairwise import _norm_winner, summarize  # noqa: E402

SYSTEM_DEFAULT = (
    "You are a strict, independent evaluator for LLM responses. "
    "Compare Response A and Response B for the same user request. "
    "Judge factual correctness, instruction following, helpfulness, and safety. "
    "Ignore length, verbosity, confidence, formatting, and model identity. "
    "If the two are genuinely equivalent, answer tie. "
    'Return JSON only: {"winner": "A" | "B" | "tie", "reason": "<=30 words"}'
)
SYSTEM_CORRECTNESS = (
    "You are a strict fact-checking evaluator. Compare Response A and Response B for the same user request. "
    "First check factual correctness and whether the answer solves the task; then instruction following; "
    "then helpfulness. Explicitly ignore length, verbosity, confidence, formatting and model identity. "
    "If both are factually wrong, answer tie. If one is factually wrong, pick the other. "
    'Return JSON only: {"winner": "A" | "B" | "tie", "reason": "<=30 words"}'
)
USER_TMPL = """User request:
{prompt}

Response A:
{a}

Response B:
{b}

Which response is better? Return JSON only."""


def post_chat(base_url, api_key, payload, timeout=120):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask(judge_cfg, prompt, a, b):
    system = SYSTEM_CORRECTNESS if judge_cfg["style"] == "correctness_first" else SYSTEM_DEFAULT
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": USER_TMPL.format(prompt=prompt, a=a, b=b)}]
    payload = {"model": judge_cfg["model"], "messages": messages,
               "temperature": judge_cfg["temperature"], "max_tokens": judge_cfg["max_tokens"]}
    payload.update(judge_cfg["extra"])
    last = None
    for attempt in range(4):
        try:
            resp = post_chat(judge_cfg["base_url"], judge_cfg["api_key"], payload)
            msg = resp["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            usage = resp.get("usage") or {}
            return _norm_winner(text), text[:500], usage
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {body}"
            if e.code in (401, 403):
                raise SystemExit(f"auth failed: {last}")
            if e.code == 400:
                raise SystemExit(f"bad request (check model/extra-json): {last}")
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = str(e)[:300]
            time.sleep(1.5 * (attempt + 1))
    return "error", f"ERROR: {last}", {}


def judge_one(cfg, p, votes, seed):
    rng = random.Random(seed)
    raw = []
    usage_tot = Counter()
    for v in range(votes):
        swap = (v % 2 == 1) if votes > 1 else False
        if votes > 2 and v >= 2:
            swap = rng.random() < 0.5
        if swap:
            w, text, usage = ask(cfg, p["prompt"], p["response_b"], p["response_a"])
            w = {"A": "B", "B": "A", "tie": "tie", "error": "error"}.get(w, "error")
        else:
            w, text, usage = ask(cfg, p["prompt"], p["response_a"], p["response_b"])
            usage_tot.update({k: v for k, v in usage.items() if isinstance(v, int)})
        raw.append({"order": "BA" if swap else "AB", "winner_original": w, "reason": text})
    cnt = Counter(x["winner_original"] for x in raw)
    if cnt.get("error", 0) == len(raw):
        final = "error"
    else:
        top = [(k, v) for k, v in cnt.most_common() if k != "error"]
        final = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else "tie"
    return {"votes": raw, "final_winner": final, "vote_counts": dict(cnt),
            "usage": dict(usage_tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="ZHIPU_API_KEY")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--prompt-style", choices=["default", "correctness_first"], default="default")
    ap.add_argument("--extra-json", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="", help="judge label in summary/model (default: model)")
    args = ap.parse_args()

    key = args.api_key or os.environ.get(args.api_key_env, "")
    if not key:
        sys.exit(f"missing API key: env {args.api_key_env} or --api-key")
    cfg = {"base_url": args.base_url, "api_key": key, "model": args.model,
           "temperature": args.temperature, "max_tokens": args.max_tokens,
           "style": args.prompt_style,
           "extra": json.loads(args.extra_json) if args.extra_json else {}}

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[data] pairs={len(pairs)} model={args.model} style={args.prompt_style} workers={args.workers}", flush=True)

    def work(i_p):
        i, p = i_p
        return i, judge_one(cfg, p, args.votes, args.seed + i)

    results = [None] * len(pairs)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for done, (i, rec) in enumerate(ex.map(work, enumerate(pairs)), 1):
            results[i] = {**pairs[i], **rec}
            if done % 50 == 0 or done == len(pairs):
                print(f"[judge] {done}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)

    errors = sum(r["final_winner"] == "error" for r in results)
    summary = summarize(results, pairs[0].get("label_a", "A"), pairs[0].get("label_b", "B"))
    summary["model"] = args.label or args.model
    summary["prompt_style"] = args.prompt_style
    summary["votes"] = args.votes
    summary["errors"] = errors
    out = {"summary": summary, "pairs": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if errors == len(results):
        sys.exit("ALL judge calls failed")


if __name__ == "__main__":
    main()
