#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RFT step 1: sample n responses per MATH prompt, keep correct ones.

Why: RFT (rejection-sampling fine-tuning / expert iteration) is the cheapest
positive-result candidate on a single 4090. It also connects to the candidate's
imitation-learning background: offline filtered BC vs on-policy GRPO.

Input: verl-format MATH parquet (prompt messages + reward_model.ground_truth),
e.g. /root/autodl-tmp/work/data/math_train_verl/train.parquet

Output:
  - train.jsonl: {"prompt": [...], "response": "...", "answer": "...", "pass_rate": k/n}
  - stats.json: yield / length / pass-rate stats

Usage:
  python scripts/rft/sample_and_filter.py \
    --model /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
    --data /root/autodl-tmp/work/data/math_train_verl/train.parquet \
    --out /root/autodl-tmp/work/data/rft_math/train.jsonl \
    --n 8 --temperature 0.7 --top-p 0.95 --max-tokens 2048
"""
import argparse
import json
import os
import re
import statistics

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


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def get_gt(ex):
    rm = ex.get("reward_model")
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return ex.get("answer") or ex.get("ground_truth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-correct", type=int, default=1)
    ap.add_argument("--prefer", choices=["shortest", "first"], default="shortest")
    args = ap.parse_args()

    import datasets
    from transformers import AutoTokenizer

    ds = datasets.load_dataset("parquet", data_files=args.data, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    rows = []
    for ex in ds:
        gt = get_gt(ex)
        if gt is None or str(gt).strip() == "":
            continue
        rows.append({"messages": ex["prompt"], "answer": str(gt).strip()})
    print(f"[data] prompts={len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(x["messages"], tokenize=False,
                                       add_generation_prompt=True) for x in rows]
    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=42)
    outs = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept, pass_rates, chosen_lens = 0, [], []
    with open(args.out, "w", encoding="utf-8") as f:
        for row, out in zip(rows, outs):
            samples = [o.text for o in out.outputs]
            preds = [extract_boxed(s) for s in samples]
            scores = [verify_one(row["answer"], p) for p in preds]
            correct_idx = [i for i, s in enumerate(scores) if s > 0]
            pass_rates.append(len(correct_idx) / len(scores))
            if len(correct_idx) < args.min_correct:
                continue
            if args.prefer == "shortest":
                idx = min(correct_idx, key=lambda i: len(samples[i].split()))
            else:
                idx = correct_idx[0]
            resp = samples[idx].strip()
            rec = {"prompt": row["messages"], "response": resp,
                   "answer": row["answer"], "pass_rate": len(correct_idx) / len(scores),
                   "chosen_words": len(resp.split())}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            chosen_lens.append(rec["chosen_words"])
            if kept % 100 == 0:
                print(f"[filter] kept={kept}", flush=True)

    stats = {
        "prompts": len(rows),
        "kept": kept,
        "yield": kept / max(len(rows), 1),
        "n": args.n,
        "avg_pass_rate": statistics.mean(pass_rates) if pass_rates else None,
        "median_pass_rate": statistics.median(pass_rates) if pass_rates else None,
        "avg_chosen_words": statistics.mean(chosen_lens) if chosen_lens else None,
        "source": args.data,
        "model": args.model,
    }
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
