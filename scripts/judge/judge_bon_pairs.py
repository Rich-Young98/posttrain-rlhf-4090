#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build N=1 vs N=64/128 pairs from best-of-N selected outputs.

Input: bon128_selected.jsonl, records with keys:
  {"N": 64, "prompt": "...", "response": "...", "proxy": ..., "gold": ..., "len": ...}

For each prompt and target N, pair:
  A = response selected at N=1 (baseline)
  B = response selected at N=target

Then run judge_pairwise.py on the produced pairs.

Example:
  python scripts/judge/judge_bon_pairs.py \
    --selected "/root/autodl-tmp/work/results/bon128_selected .jsonl" \
    --ns 64 128 \
    --out /root/autodl-tmp/work/results/bon_pairs.jsonl
  python scripts/judge/judge_pairwise.py \
    --pairs /root/autodl-tmp/work/results/bon_pairs.jsonl \
    --out /root/autodl-tmp/work/results/bon_judge.json --backend api
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    by_prompt = defaultdict(dict)  # prompt -> {N: record}
    with open(args.selected, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = " ".join(str(d["prompt"]).split())
            by_prompt[p][int(d["N"])] = d

    pairs = []
    for target in args.ns:
        for prompt, ns in by_prompt.items():
            if 1 not in ns or target not in ns:
                continue
            a = ns[1]
            b = ns[target]
            if a["response"].strip() == b["response"].strip():
                continue
            pairs.append({
                "id": f"bon-{target}-{len(pairs):04d}",
                "prompt": prompt,
                "response_a": a["response"],
                "response_b": b["response"],
                "label_a": "N=1",
                "label_b": f"N={target}",
                "meta": {"target_N": target, "len_a": a.get("len"),
                         "len_b": b.get("len"),
                         "proxy_a": a.get("proxy"), "proxy_b": b.get("proxy"),
                         "gold_a": a.get("gold"), "gold_b": b.get("gold")},
            })
    if args.limit:
        pairs = pairs[: args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[done] pairs={len(pairs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
