#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched vLLM version of generate_pair_outputs.py (base vs PEFT adapter).

Generates both sides with one vLLM engine: base pass + LoRA pass, so 566 prompts
take a couple of minutes instead of hours.

Usage:
  python scripts/judge/generate_pair_outputs_vllm.py \
    --base_model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    --adapter /root/autodl-tmp/work/runs/dpo-b005/final \
    --prompts_parquet /root/autodl-tmp/work/data/uf_rm/holdout.parquet \
    --n 566 --seed 43 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
    --out /root/autodl-tmp/work/results/judge_v2/dpo800_pairs_566.jsonl \
    --label_a base --label_b dpo800
"""
import argparse
import json
import random

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def load_prompts(args):
    prompts = []
    if args.prompts_jsonl:
        for line in open(args.prompts_jsonl, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                prompts.append(as_text(d.get("prompt", d.get("question", d))))
    else:
        import datasets
        ds = datasets.load_dataset("parquet", data_files=args.prompts_parquet, split="train")
        for x in ds:
            prompts.append(as_text(x.get("prompt", x.get("question", ""))))
    seen, uniq = set(), []
    for p in prompts:
        p = " ".join(p.split())
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    random.Random(args.seed).shuffle(uniq)
    return uniq[: args.n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts_jsonl", default=None)
    ap.add_argument("--prompts_parquet", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--label_a", default="base")
    ap.add_argument("--label_b", default="adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    prompts = load_prompts(args)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True) for p in prompts]
    print(f"[data] prompts={len(prompts)} base={args.base_model} adapter={args.adapter}", flush=True)

    llm = LLM(model=args.base_model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              max_loras=1, enforce_eager=True)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, seed=args.seed)
    print("[gen] base", flush=True)
    base_outs = llm.generate(chat, sp)
    print("[gen] adapter", flush=True)
    lora_outs = llm.generate(chat, sp, lora_request=LoRARequest("adapter", 1, args.adapter))

    with open(args.out, "w", encoding="utf-8") as f:
        for i, (p, bo, lo) in enumerate(zip(prompts, base_outs, lora_outs)):
            rec = {"id": f"pair-{i:04d}", "prompt": p,
                   "response_a": bo.outputs[0].text.strip(),
                   "response_b": lo.outputs[0].text.strip(),
                   "label_a": args.label_a, "label_b": args.label_b}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] {len(prompts)} pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
