#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a math model with vLLM: avg@k / pass@k / length, under a frozen protocol.

2026-10-26 v2 (Day 2 unified protocol). Changes vs the original eval_math.py:
  1. --prompt-mode {legacy,bare}
       legacy: d2_tools.py templates (MATH-500 asks for \\boxed{}; AIME asks for "Answer: ...")
       bare:   raw problem only (old eval_math.py behaviour)
  2. --parser {v2,legacy}
       v2:     last balanced \\boxed{...} -> "Answer:/final answer" line -> None.
               NO bare-number fallback, so the main score never depends on it.
       legacy: old eval_math.py behaviour (adds last-number fallback) for traceability only.
  3. --dump-jsonl: one record per (seed, question, sample), see FIELDS below.
  4. --run-id / --evaluator-version recorded in every record and in the summary.

Metrics are unchanged in meaning: avg@n (mean sample accuracy), pass@1 and pass@k
(unbiased pass@k estimator), avg_words, plus parse/truncation diagnostics.

Usage:
  python scripts/eval/eval_math.py \
    --model /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
    --data /root/autodl-tmp/work/data/eval/math500.jsonl \
    --n 4 --seeds 42,43,44 --prompt-mode legacy --parser v2 \
    --max-new-tokens 3072 --out results/base_math500.json \
    --dump-jsonl results/base_math500.jsonl --run-id p1-base-unified
"""
import argparse
import functools
import hashlib
import json
import os
import re
import signal
import statistics
from math import comb

from vllm import LLM, SamplingParams

EVALUATOR_VERSION = "p1-eval-v2.0-20261026"

AIME_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    "{problem}"
)
MATH_TEMPLATE = (
    "Solve the following math problem step by step. Put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)

ANSWER_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


# ---------------------------------------------------------------- parsers
def last_boxed(text: str):
    """Return the content of the last balanced \\boxed{...}, or None."""
    idx = text.rfind("\\boxed")
    while idx != -1:
        i = idx + len("\\boxed")
        while i < len(text) and text[i] in " \t\n":
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1:j].strip()
                j += 1
        idx = text.rfind("\\boxed", 0, idx)
    return None


def extract_pred(text: str, parser: str):
    """Return (pred, parse_mode). parse_mode is boxed | answer_regex | number_fallback | none."""
    text = text or ""
    boxed = last_boxed(text)
    if boxed:
        return boxed, "boxed"
    m = ANSWER_RE.findall(text)
    if m:
        pred = m[-1].strip().rstrip(".")
        if pred:
            return pred, "answer_regex"
    if parser == "legacy":
        nums = NUMBER_RE.findall(text)
        if nums:
            return nums[-1], "number_fallback"
    return None, "none"


class _VerifyTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _VerifyTimeout()


_GOLD_CFG = None
_PRED_CFG = None


def _cfgs():
    global _GOLD_CFG, _PRED_CFG
    if _GOLD_CFG is None:
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        _GOLD_CFG = (LatexExtractionConfig(),)
        _PRED_CFG = (ExprExtractionConfig(), LatexExtractionConfig())
    return _GOLD_CFG, _PRED_CFG


@functools.lru_cache(maxsize=200000)
def _verify_cached(gold: str, pred: str) -> float:
    r"""math_verify in the verl-compatible configuration.

    The default math_verify.parse() is too weak: it drops LaTeX tuples
    (\left(3, \frac{\pi}{2}\right)) and symbolic answers (p - q). verl wraps both
    sides in \boxed{} and parses with (LatexExtractionConfig,) / (Expr+Latex) —
    that handles tuples, symbolic expressions and 90^\circ vs 90 correctly.
    """
    from math_verify.grader import verify
    from math_verify.parser import parse

    gold_cfg, pred_cfg = _cfgs()
    g = parse("\\boxed{" + gold + "}", gold_cfg)
    p = parse("\\boxed{" + pred + "}", pred_cfg)
    if g and p:
        return 1.0 if any(verify(gg, pp) for gg in g for pp in p) else 0.0
    return 0.0


def verify_one(gold: str, pred: str):
    """Return (score, error). score in {0.0, 1.0}."""
    if pred is None:
        return 0.0, ""
    prev = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        return _verify_cached(str(gold), str(pred)), ""
    except _VerifyTimeout:
        return 0.0, "verify_timeout"
    except Exception as exc:  # noqa: BLE001
        return 0.0, str(exc)[:160]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------- helpers
def benchmark_of(path: str, rows) -> str:
    base = os.path.basename(path).lower()
    for name in ("math500", "aime24", "aime25"):
        if name in base:
            return name
    if rows and str(rows[0].get("id", "")).startswith("aime24"):
        return "aime24"
    if rows and str(rows[0].get("id", "")).startswith("aime25"):
        return "aime25"
    return "math500"


def build_prompt(tok, problem: str, benchmark: str, prompt_mode: str) -> str:
    if prompt_mode == "legacy":
        template = MATH_TEMPLATE if benchmark == "math500" else AIME_TEMPLATE
        content = template.format(problem=problem)
    else:
        content = problem
    return tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-mode", choices=["legacy", "bare"], default="legacy")
    ap.add_argument("--parser", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--dump-jsonl", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    args = ap.parse_args()

    data = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if args.limit:
        data = data[: args.limit]
    benchmark = benchmark_of(args.data, data)
    print(f"[data] {len(data)} problems benchmark={benchmark} n={args.n} "
          f"prompt_mode={args.prompt_mode} parser={args.parser}", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    prompts = [build_prompt(tok, x["problem"], benchmark, args.prompt_mode) for x in data]
    prompt_hashes = [hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts]
    model_id = args.model.rstrip("/").split("/")[-1]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    per_seed, all_problems = [], []
    jsonl_f = open(args.dump_jsonl, "w", encoding="utf-8") if args.dump_jsonl else None
    if jsonl_f:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_jsonl)), exist_ok=True)

    for seed in seeds:
        sp = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=seed,
        )
        outs = llm.generate(prompts, sp)
        problems = []
        for qi, (ex, out) in enumerate(zip(data, outs)):
            samples = [o.text for o in out.outputs]
            finish = [getattr(o, "finish_reason", None) for o in out.outputs]
            preds, modes, scores, errs, lens, toks, truncs = [], [], [], [], [], [], []
            for si, text in enumerate(samples):
                pred, mode = extract_pred(text, args.parser)
                score, err = verify_one(ex["answer"], pred)
                ntok = len(tok(text, add_special_tokens=False)["input_ids"])
                truncated = finish[si] == "length"
                preds.append(pred)
                modes.append(mode)
                scores.append(score)
                errs.append(err)
                lens.append(len(text.split()))
                toks.append(ntok)
                truncs.append(truncated)
                if jsonl_f:
                    jsonl_f.write(json.dumps({
                        "run_id": args.run_id or model_id,
                        "model_id": model_id,
                        "benchmark": benchmark,
                        "question_id": ex.get("id"),
                        "eval_seed": seed,
                        "sample_idx": si,
                        "prompt_mode": args.prompt_mode,
                        "parser": args.parser,
                        "prompt_hash": prompt_hashes[qi],
                        "completion": text,
                        "parsed_answer": pred,
                        "parse_mode": mode,
                        "reference": str(ex["answer"]),
                        "correct": score,
                        "token_count": ntok,
                        "truncated": truncated,
                        "parse_error": err,
                        "evaluator_version": EVALUATOR_VERSION,
                    }, ensure_ascii=False) + "\n")
            problems.append({
                "id": ex.get("id"),
                "gold": str(ex["answer"]),
                "preds": preds,
                "scores": scores,
                "avg": sum(scores) / len(scores),
                "lens": lens,
                "token_counts": toks,
                "truncated": truncs,
                "parse_modes": modes,
            })

        n = args.n
        present = sum(1 for p in problems if max(p["scores"]) > 0.5)
        avg_at_n = sum(p["avg"] for p in problems) / len(problems)
        pass1 = sum(pass_at_k(n, int(sum(p["scores"])), 1) for p in problems) / len(problems)
        k = min(8, n)
        pk = sum(pass_at_k(n, int(sum(p["scores"])), k) for p in problems) / len(problems)
        avg_words = sum(sum(p["lens"]) / len(p["lens"]) for p in problems) / len(problems)
        avg_tokens = sum(sum(p["token_counts"]) / len(p["token_counts"]) for p in problems) / len(problems)
        n_samples = n * len(problems)
        parse_fail = sum(1 for p in problems for m in p["parse_modes"] if m == "none") / n_samples
        fallback = sum(1 for p in problems for m in p["parse_modes"] if m == "number_fallback") / n_samples
        trunc_rate = sum(1 for p in problems for t in p["truncated"] if t) / n_samples
        summary = {
            "seed": seed,
            "n": n,
            "avg@n": avg_at_n,
            "pass@1": pass1,
            f"pass@{k}": pk,
            "avg_words": avg_words,
            "avg_tokens": avg_tokens,
            "correct_fraction": sum(sum(p["scores"]) for p in problems) / n_samples,
            "parse_fail_rate": parse_fail,
            "number_fallback_rate": fallback,
            "truncation_rate": trunc_rate,
            "problems_with_any_correct": present,
        }
        per_seed.append(summary)
        all_problems.append({"seed": seed, "problems": problems})
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    if jsonl_f:
        jsonl_f.close()

    keys = ["avg@n", "pass@1", f"pass@{min(8, args.n)}", "avg_words", "avg_tokens",
            "correct_fraction", "parse_fail_rate", "number_fallback_rate", "truncation_rate"]
    agg = {k: {"mean": statistics.mean(s[k] for s in per_seed),
               "std": statistics.pstdev([s[k] for s in per_seed])} for k in keys}
    result = {
        "model": args.model,
        "model_id": model_id,
        "data": args.data,
        "benchmark": benchmark,
        "run_id": args.run_id,
        "n": args.n,
        "seeds": seeds,
        "prompt_mode": args.prompt_mode,
        "parser": args.parser,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "evaluator_version": EVALUATOR_VERSION,
        "per_seed": per_seed,
        "aggregate": agg,
        "problems": all_problems,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print("[aggregate]", json.dumps(agg, ensure_ascii=False), flush=True)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
