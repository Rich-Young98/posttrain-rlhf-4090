#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paired outputs (base vs adapter) for independent judging.

Example (DPO 800):
  python scripts/judge/generate_pair_outputs.py \
    --base_model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    --adapter /root/autodl-tmp/work/runs/dpo-b005/final \
    --prompts_parquet /root/autodl-tmp/work/data/uf_rm/holdout.parquet \
    --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
    --out /root/autodl-tmp/work/results/dpo800_pairs.jsonl \
    --label_a base --label_b dpo800

Then:
  python scripts/judge/judge_pairwise.py --pairs ... --out ... --backend api
"""
import argparse
import json
import os
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    prompts.append(as_text(d.get("prompt", d.get("question", d))))
    elif args.prompts_parquet:
        import datasets

        ds = datasets.load_dataset("parquet", data_files=args.prompts_parquet, split="train")
        for x in ds:
            prompts.append(as_text(x.get("prompt", x.get("question", ""))))
    else:
        raise SystemExit("need --prompts_jsonl or --prompts_parquet")
    seen, uniq = set(), []
    for p in prompts:
        p = " ".join(p.split())
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    random.Random(args.seed).shuffle(uniq)
    return uniq[: args.n]


@torch.no_grad()
def generate(model, tok, prompt, args):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(args.device)
    out = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
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
    args = ap.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args)
    print(f"[data] prompts={len(prompts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    print(f"[model] {args.base_model} + {args.adapter}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            # same seed for both sides -> comparable sampling
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            with model.disable_adapter():
                resp_a = generate(model, tok, prompt, args)
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            resp_b = generate(model, tok, prompt, args)
            rec = {"id": f"pair-{i:04d}", "prompt": prompt,
                   "response_a": resp_a, "response_b": resp_b,
                   "label_a": args.label_a, "label_b": args.label_b}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                print(f"[gen] {i+1}/{len(prompts)}", flush=True)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
