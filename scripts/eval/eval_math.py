#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a math model with vLLM: avg@k / pass@k / length.

Usage:
  python scripts/eval/eval_math.py \
    --model /root/autodl-tmp/work/runs/rft-merged \
    --data /root/autodl-tmp/work/data/eval/math500.jsonl \
    --out /root/autodl-tmp/work/results/rft_math500.json \
    --n 4 --seeds 42,43,44 --max-new-tokens 2048
"""
import argparse
import json
import os
import re
import statistics
from math import comb

from vllm import LLM, SamplingParams


def extract_boxed(text: str):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n]+)", text or "", re.I)
    if m:
        return m[-1].strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return m[-1] if m else None


def verify_one(gold: str, pred: str) -> float:
    if pred is None:
        return 0.0
    try:
        from math_verify import parse, verify

        return float(verify(parse(gold), parse(pred)))
    except Exception:
        return 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if args.limit:
        data = data[: args.limit]
    print(f"[data] {len(data)} problems; n={args.n}", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": x["problem"]}],
                                       tokenize=False, add_generation_prompt=True)
               for x in data]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    per_seed = []
    all_problems = []
    for seed in seeds:
        sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_new_tokens, seed=seed)
        outs = llm.generate(prompts, sp)
        problems = []
        for ex, out in zip(data, outs):
            samples = [o.text for o in out.outputs]
            preds = [extract_boxed(s) for s in samples]
            scores = [verify_one(ex["answer"], p) for p in preds]
            problems.append({"id": ex["id"], "gold": ex["answer"], "preds": preds,
                             "scores": scores, "avg": sum(scores) / len(scores),
                             "lens": [len(s.split()) for s in samples]})
        n = args.n
        avg_at_n = sum(p["avg"] for p in problems) / len(problems)
        c_total = sum(sum(p["scores"]) for p in problems)
        pass_all = sum(pass_at_k(n, int(sum(p["scores"])), n) for p in problems) / len(problems)
        avg_len = sum(sum(p["lens"]) / len(p["lens"]) for p in problems) / len(problems)
        summary = {"seed": seed, "n": n, "avg@n": avg_at_n,
                   "pass@n": pass_all, "avg_words": avg_len,
                   "correct_fraction": c_total / (n * len(problems))}
        per_seed.append(summary)
        all_problems.append({"seed": seed, "problems": problems})
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    keys = ["avg@n", "pass@n", "avg_words", "correct_fraction"]
    agg = {k: {"mean": statistics.mean(s[k] for s in per_seed),
               "std": statistics.pstdev([s[k] for s in per_seed])} for k in keys}
    result = {"model": args.model, "data": args.data, "n": args.n,
              "seeds": seeds, "per_seed": per_seed, "aggregate": agg,
              "problems": all_problems}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("[aggregate]", json.dumps(agg, ensure_ascii=False), flush=True)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
