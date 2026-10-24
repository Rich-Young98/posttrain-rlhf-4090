#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D7 复核：RewardBench 全量 + 长度/模型身份分组分析。

回答三个问题：
1) 500 条 80.8% 是不是抽样偶然？-> 跑全量 2985。
2) 80.8% 是不是长度捷径？-> 长度分桶 acc、length-only baseline、score-length 相关。
3) 80.8% 是不是“认模型风格”？-> same_model / human-human / GPT-4 chosen 分组 acc。

用法：
  /root/autodl-tmp/venv-verl/bin/python scripts/analyze_rm.py \
    --adapter /root/autodl-tmp/work/runs/rm-lora/final --n 2985
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import datasets
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
RB_DIR = "/root/autodl-tmp/work/data/raw/rewardbench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/root/autodl-tmp/work/runs/rm-lora/final")
    ap.add_argument("--n", type=int, default=2985, help="0 表示全量")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out", default="/root/autodl-tmp/work/results/rm_full_analysis.json")
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
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    print("[model]", args.adapter)

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

    def ntok(s):
        return len(tok(as_text(s), add_special_tokens=False)["input_ids"])

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
        return model(**enc).logits.view(-1).float().cpu().numpy()

    files = [
        f
        for f in sorted(glob.glob(os.path.join(RB_DIR, "**", "*.parquet"), recursive=True))
        if "filtered" in os.path.basename(f)
    ]
    if files:
        ds = datasets.load_dataset("parquet", data_files=files, split="train")
    else:
        ds = datasets.load_dataset("allenai/reward-bench", split="filtered")
    if args.n and args.n < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.n))
    print("[data]", len(ds))

    cs, rs, cl, rl, same, subs, chosen_models = [], [], [], [], [], [], []
    for i in range(0, len(ds), args.bs):
        b = ds[i : i + args.bs]
        cs.append(batch_score(b["prompt"], b["chosen"]))
        rs.append(batch_score(b["prompt"], b["rejected"]))
        cl.extend([ntok(x) for x in b["chosen"]])
        rl.extend([ntok(x) for x in b["rejected"]])
        same.extend([str(a) == str(b_) for a, b_ in zip(b["chosen_model"], b["rejected_model"])])
        subs.extend(b["subset"])
        chosen_models.extend(b["chosen_model"])
        if (i // args.bs) % 20 == 0:
            print(f"[score] {min(i + args.bs, len(ds))}/{len(ds)}", flush=True)

    cs = np.concatenate(cs).astype(np.float64)
    rs = np.concatenate(rs).astype(np.float64)
    cl = np.asarray(cl)
    rl = np.asarray(rl)
    same = np.asarray(same)
    subs = np.asarray(subs)
    chosen_models = np.asarray(chosen_models)
    diff = cl - rl
    gap = cs - rs
    correct = cs > rs

    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=np.float64)
        ranks[order] = np.arange(len(a))
        return ranks

    def summarize(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "rm_acc": round(float(correct[mask].mean()), 4),
            "length_only_acc": round(float((diff[mask] > 0).mean()), 4),
            "mean_len_diff": round(float(diff[mask].mean()), 1),
            "mean_score_gap": round(float(gap[mask].mean()), 4),
        }

    result = {
        "adapter": args.adapter,
        "n": int(len(ds)),
        "overall": {
            "rm_acc": round(float(correct.mean()), 4),
            "length_only_acc": round(float((diff > 0).mean()), 4),
            "mean_len_diff": round(float(diff.mean()), 1),
            "mean_score_gap": round(float(gap.mean()), 4),
        },
        "pearson_len_diff_vs_score_gap": round(float(np.corrcoef(diff, gap)[0, 1]), 4),
        "spearman_len_diff_vs_score_gap": round(
            float(np.corrcoef(rankdata(diff), rankdata(gap))[0, 1]), 4
        ),
        "score_len_corr_all_responses": round(
            float(np.corrcoef(np.concatenate([cl, rl]), np.concatenate([cs, rs]))[0, 1]), 4
        ),
        "groups": {
            "same_model": summarize(same),
            "different_model": summarize(~same),
        },
        "length_bins": {},
        "by_subset": {},
    }

    # 长度差分桶：看“chosen 更长/更短/长度相近”时 RM 是否仍然准
    bins = [
        ("chosen_much_shorter", -10**9, -200),
        ("chosen_shorter", -200, -50),
        ("similar", -50, 50),
        ("chosen_longer", 50, 200),
        ("chosen_much_longer", 200, 10**9),
    ]
    for name, lo, hi in bins:
        result["length_bins"][name] = summarize((diff > lo) & (diff <= hi))

    # 特殊小分组：human-human / GPT-4 chosen / 长度几乎相同
    human_rej = np.asarray([str(x) == "human" for x in ds["rejected_model"]])
    gpt4_chosen = np.isin(chosen_models, ["GPT-4", "GPT4-Turbo"])
    result["groups"]["human_human"] = summarize(
        (chosen_models == "human") & human_rej
    )
    result["groups"]["gpt4_chosen"] = summarize(gpt4_chosen)
    result["groups"]["length_matched_20"] = summarize(np.abs(diff) <= 20)

    for sub in sorted(set(subs)):
        result["by_subset"][str(sub)] = summarize(subs == sub)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[overall]", json.dumps(result["overall"], ensure_ascii=False))
    print("[corr] pearson", result["pearson_len_diff_vs_score_gap"],
          "spearman", result["spearman_len_diff_vs_score_gap"])
    print("[length_bins]", json.dumps(result["length_bins"], ensure_ascii=False, indent=2))
    print("[groups]", json.dumps(result["groups"], ensure_ascii=False, indent=2))
    print("[done] ->", args.out)


if __name__ == "__main__":
    main()
