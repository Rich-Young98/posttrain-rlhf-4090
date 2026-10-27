#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a blind human-eval sheet from judge pair JSONs.

Samples pairs per experiment, randomizes A/B order (hidden in output), and writes
a CSV + Markdown table with empty verdict columns.

Usage:
  python scripts/judge/make_human_sheet.py --dir p2/judge_v2 \
    --spec dpo800=60 dpo300=40 N=64=25 N=128=25 --out p2/human_150 --seed 42
"""
import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="exp=n e.g. dpo800=60")
    ap.add_argument("--out", required=True, help="output prefix (csv + md)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-file", default="", help="judge JSON to take pairs from (optional)")
    args = ap.parse_args()

    pairs = {}
    if args.judge_file:
        d = json.load(open(args.judge_file, encoding="utf-8"))
        for r in d["pairs"]:
            pairs.setdefault(r.get("label_b"), []).append(r)
    else:
        import glob
        for f in glob.glob(os.path.join(args.dir, "*.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for r in d.get("pairs", []):
                exp = r.get("label_b")
                if exp and exp not in pairs:
                    pairs[exp] = d["pairs"]

    rng = random.Random(args.seed)
    rows = []
    for item in args.spec:
        exp, n = item.rsplit("=", 1)
        n = int(n)
        pool = pairs.get(exp, [])
        rng.shuffle(pool)
        for r in pool[:n]:
            # randomize which side is shown first
            swap = rng.random() < 0.5
            rows.append({
                "id": r.get("id", ""), "experiment": exp,
                "shown_a": r["response_b"] if swap else r["response_a"],
                "shown_b": r["response_a"] if swap else r["response_b"],
                "order": "BA" if swap else "AB",
                "prompt": r["prompt"],
                "human_winner": "", "human_notes": "",
                "human_fact": "", "human_follow": "", "human_reason": "", "human_help": "",
            })
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_path = args.out + ".csv"
    md_path = args.out + ".md"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 人工盲评表（150 对）\n\n")
        f.write("> 隐藏来源；A/B 顺序随机。评分：1-5（事实/遵循/推理/有用），winner 填 A/B/tie。\n\n")
        for i, r in enumerate(rows, 1):
            f.write(f"## {i}. [{r['experiment']}] {r['id']}\n\n")
            f.write(f"**Prompt**: {r['prompt'][:800]}\n\n")
            f.write(f"**A**: {r['shown_a'][:1200]}\n\n")
            f.write(f"**B**: {r['shown_b'][:1200]}\n\n")
            f.write("winner: ____  fact: _  follow: _  reason: _  help: _  notes: ____\n\n---\n\n")
    print(f"[done] {len(rows)} pairs -> {csv_path} / {md_path}")


if __name__ == "__main__":
    main()
