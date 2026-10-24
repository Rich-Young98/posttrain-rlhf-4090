#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D6：评 LoRA RM。

1) RewardBench filtered 子集（默认 500 条，打乱后抽样）；
2) 可选：UltraFeedback 留出集 300 pair 的 in-distribution 准确率。

用法：
  /root/autodl-tmp/venv-verl/bin/python /root/autodl-tmp/work/scripts/eval_rm.py \
    --adapter /root/autodl-tmp/work/runs/rm-lora/final --n 500
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import datasets
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/root/autodl-tmp/work/runs/rm-lora/final")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--no_adapter", action="store_true", help="只评 base 模型，做 sanity check")
    ap.add_argument("--uf_holdout", default="/root/autodl-tmp/work/data/uf_rm/holdout.parquet")
    ap.add_argument("--uf_pairs", type=int, default=300)
    ap.add_argument("--out", default="/root/autodl-tmp/work/results/rm_eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=1, torch_dtype=torch.bfloat16
    )
    base.config.pad_token_id = tok.pad_token_id
    if args.no_adapter:
        model = base
        print("[model] base only（没有加载 adapter）")
    else:
        if not os.path.isdir(args.adapter):
            raise SystemExit(f"adapter 不存在：{args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter)
        print(f"[model] base + adapter: {args.adapter}")
    model = model.to(device).eval()

    def as_text(x):
        if isinstance(x, list):
            if x and isinstance(x[0], dict):
                return " ".join(str(m.get("content", "")) for m in x)
            return " ".join(str(t) for t in x)
        if isinstance(x, dict):
            return str(x.get("content", ""))
        return str(x)

    def render(prompt, response):
        msgs = [
            {"role": "user", "content": as_text(prompt)},
            {"role": "assistant", "content": as_text(response)},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    @torch.no_grad()
    def batch_score(prompts, responses):
        texts = [render(p, r) for p, r in zip(prompts, responses)]
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        ).to(device)
        return model(**enc).logits.view(-1).float().cpu().tolist()

    def pair_acc(ds, tag):
        total = defaultdict(int)
        correct = defaultdict(int)
        for i in range(0, len(ds), args.bs):
            b = ds[i : i + args.bs]
            cs = batch_score(b["prompt"], b["chosen"])
            rs = batch_score(b["prompt"], b["rejected"])
            subs = b.get("subset", ["all"] * len(cs))
            for sub, c, r in zip(subs, cs, rs):
                total[sub] += 1
                correct[sub] += int(c > r)
            if (i // args.bs) % 10 == 0:
                done = min(i + args.bs, len(ds))
                print(f"[{tag}] {done}/{len(ds)}", flush=True)
        acc = sum(correct.values()) / max(sum(total.values()), 1)
        by_subset = {
            k: {"n": total[k], "acc": round(correct[k] / total[k], 4)}
            for k in sorted(total)
        }
        print(f"[{tag}] acc = {acc:.4f}  n = {sum(total.values())}")
        print(f"[{tag}] by_subset = {json.dumps(by_subset, ensure_ascii=False)}")
        return acc, by_subset

    print("[data] 加载 RewardBench filtered ...")
    local_files = [
        f
        for f in sorted(
            glob.glob(
                "/root/autodl-tmp/work/data/raw/rewardbench/**/*.parquet",
                recursive=True,
            )
        )
        if "filtered" in os.path.basename(f)
    ]
    if local_files:
        print(f"[data] 本地 parquet: {local_files}")
        rb = datasets.load_dataset("parquet", data_files=local_files, split="train")
    else:
        rb = datasets.load_dataset("allenai/reward-bench", split="filtered")
    if args.n and args.n < len(rb):
        rb = rb.shuffle(seed=0).select(range(args.n))
    rb_acc, rb_by_subset = pair_acc(rb, "rewardbench")

    result = {
        "adapter": None if args.no_adapter else args.adapter,
        "rewardbench": {"n": len(rb), "acc": round(rb_acc, 4), "by_subset": rb_by_subset},
    }

    if args.uf_pairs > 0 and os.path.exists(args.uf_holdout):
        uf = datasets.load_dataset("parquet", data_files=args.uf_holdout, split="train")
        uf = uf.select(range(min(args.uf_pairs, len(uf))))
        uf_acc, _ = pair_acc(uf, "uf_holdout")
        result["uf_holdout"] = {"n": len(uf), "acc": round(uf_acc, 4)}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
