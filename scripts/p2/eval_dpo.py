#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D8 生成评测：base vs DPO + RM proxy + 长度分桶。

用法：
  5 条 sanity：
    python scripts/eval_dpo.py --n 5 --print_n 5 \
      --out /root/autodl-tmp/work/results/dpo_sanity5.jsonl

  200 条正式：
    python scripts/eval_dpo.py --n 200 --print_n 0 \
      --out /root/autodl-tmp/work/results/dpo_eval_200.jsonl \
      --report /root/autodl-tmp/work/results/dpo_eval_200.json

注意：RM proxy 是在 UltraFeedback 上训练的，和 DPO 同分布，且有长度偏置；
它只能当 proxy，不能当独立 judge。最终结论要配合人工 20 条和长度分桶。
"""
import argparse
import json
import os
import random

import datasets
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
HOLDOUT = "/root/autodl-tmp/work/data/uf_rm/holdout.parquet"


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def load_prompts(n, seed):
    ds = datasets.load_dataset("parquet", data_files=HOLDOUT, split="train")
    seen, prompts = set(), []
    for x in ds:
        p = " ".join(as_text(x["prompt"]).split())
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
    random.Random(seed).shuffle(prompts)
    return prompts[: min(n, len(prompts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/root/autodl-tmp/work/runs/dpo-b005/final")
    ap.add_argument("--rm_adapter", default="/root/autodl-tmp/work/runs/rm-lora/final")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--print_n", type=int, default=5)
    ap.add_argument("--out", default="/root/autodl-tmp/work/results/dpo_eval.jsonl")
    ap.add_argument("--report", default="/root/autodl-tmp/work/results/dpo_eval_report.json")
    ap.add_argument("--manual_md", default="/root/autodl-tmp/work/results/dpo_manual_review_20.md")
    ap.add_argument("--no_rm", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args.n, args.seed)
    print("[data] prompts", len(prompts), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # 生成时左 padding 更稳

    # 一份 base 权重，外面挂 DPO LoRA；base 生成时 disable_adapter()
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    dpo = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    print("[model] base + DPO adapter:", args.adapter, flush=True)

    @torch.no_grad()
    def generate(prompt, use_adapter):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        input_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if use_adapter:
            out = dpo.generate(**enc, **gen_kwargs)
        else:
            with dpo.disable_adapter():
                out = dpo.generate(**enc, **gen_kwargs)
        new_ids = out[0, input_len:]
        return tok.decode(new_ids, skip_special_tokens=True).strip()

    records = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        base_resp = generate(prompt, use_adapter=False)
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        dpo_resp = generate(prompt, use_adapter=True)
        base_len = len(tok(base_resp, add_special_tokens=False)["input_ids"])
        dpo_len = len(tok(dpo_resp, add_special_tokens=False)["input_ids"])
        records.append({
            "i": i,
            "prompt": prompt,
            "base": base_resp,
            "dpo": dpo_resp,
            "base_len": base_len,
            "dpo_len": dpo_len,
            "len_diff": dpo_len - base_len,
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[gen] {i + 1}/{len(prompts)}", flush=True)

    # RM proxy：只作为 proxy，不是独立 judge
    if not args.no_rm:
        rm_tok = AutoTokenizer.from_pretrained(BASE)
        if rm_tok.pad_token is None:
            rm_tok.pad_token = rm_tok.eos_token
        rm_tok.padding_side = "right"
        rm_base = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=1, torch_dtype=torch.bfloat16
        )
        rm_base.config.pad_token_id = rm_tok.pad_token_id
        rm = PeftModel.from_pretrained(rm_base, args.rm_adapter).to(device).eval()
        print("[model] RM proxy:", args.rm_adapter, flush=True)

        @torch.no_grad()
        def rm_score(prompt, response):
            text = rm_tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
            enc = rm_tok(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)
            return float(rm(**enc).logits.view(-1)[0].item())

        for i, r in enumerate(records):
            r["rm_base"] = rm_score(r["prompt"], r["base"])
            r["rm_dpo"] = rm_score(r["prompt"], r["dpo"])
            r["rm_gap"] = r["rm_dpo"] - r["rm_base"]
            if (i + 1) % 20 == 0 or i == 0:
                print(f"[rm] {i + 1}/{len(records)}", flush=True)

    # 汇总
    result = {
        "n": len(records),
        "judge": "RM proxy (UltraFeedback-trained, length-biased; not independent)",
        "adapter": args.adapter,
        "rm_adapter": None if args.no_rm else args.rm_adapter,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_len_mean": round(sum(r["base_len"] for r in records) / max(len(records), 1), 2),
        "dpo_len_mean": round(sum(r["dpo_len"] for r in records) / max(len(records), 1), 2),
        "len_diff_mean": round(sum(r["len_diff"] for r in records) / max(len(records), 1), 2),
    }
    if not args.no_rm:
        wins = sum(1 for r in records if r["rm_dpo"] > r["rm_base"])
        ties = sum(1 for r in records if r["rm_dpo"] == r["rm_base"])
        result["rm_win_rate"] = round((wins + 0.5 * ties) / max(len(records), 1), 4)
        result["rm_ties"] = ties
        result["rm_gap_mean"] = round(
            sum(r["rm_gap"] for r in records) / max(len(records), 1), 4
        )
        bins = [
            ("dpo_shorter", -10**9, -20),
            ("similar", -20, 20),
            ("dpo_longer", 20, 10**9),
        ]
        result["length_bins"] = {}
        for name, lo, hi in bins:
            sub = [r for r in records if lo < r["len_diff"] <= hi]
            if not sub:
                result["length_bins"][name] = {"n": 0}
                continue
            w = sum(1 for r in sub if r["rm_dpo"] > r["rm_base"])
            t = sum(1 for r in sub if r["rm_dpo"] == r["rm_base"])
            result["length_bins"][name] = {
                "n": len(sub),
                "rm_win_rate": round((w + 0.5 * t) / len(sub), 4),
                "mean_len_diff": round(sum(r["len_diff"] for r in sub) / len(sub), 1),
            }

    for path in (args.out, args.report, args.manual_md):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 人工 20 条
    review = random.Random(0).sample(records, min(20, len(records)))
    with open(args.manual_md, "w", encoding="utf-8") as f:
        f.write("# DPO 人工复核 20 条\n\n")
        for j, r in enumerate(review, 1):
            f.write(f"## {j}. {r['prompt'][:120]}\n\n")
            f.write(f"**Base** (len={r['base_len']})\n\n{r['base']}\n\n")
            f.write(f"**DPO** (len={r['dpo_len']})\n\n{r['dpo']}\n\n")
            if "rm_base" in r:
                f.write(f"RM: base={r['rm_base']:.4f}, dpo={r['rm_dpo']:.4f}, gap={r['rm_gap']:.4f}\n\n")
            f.write("---\n\n")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[result]", json.dumps(result, ensure_ascii=False, indent=2))
    print("[done] jsonl:", args.out)
    print("[done] report:", args.report)
    print("[done] manual:", args.manual_md)

    for r in records[: args.print_n]:
        print("\n=== PROMPT ===")
        print(r["prompt"][:300])
        print("--- BASE ---")
        print(r["base"][:800])
        print("--- DPO ---")
        print(r["dpo"][:800])
        if "rm_base" in r:
            print(f"RM base={r['rm_base']:.4f} dpo={r['rm_dpo']:.4f} gap={r['rm_gap']:.4f}")


if __name__ == "__main__":
    main()
