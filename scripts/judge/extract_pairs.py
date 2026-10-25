#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover pair JSONL from an existing judge JSON (no GPU generation needed).

Use when *_judge.json exists but the judge calls failed (e.g. 401) and the
original *_pairs.jsonl is not on this machine.

Usage:
  python scripts/judge/extract_pairs.py \
    --judge_json p2/dpo300_judge.json \
    --out /root/autodl-tmp/work/results/dpo300_pairs.jsonl
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--judge_json", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

d = json.load(open(args.judge_json, encoding="utf-8"))
pairs = d.get("pairs", [])
with open(args.out, "w", encoding="utf-8") as f:
    for p in pairs:
        rec = {"id": p.get("id"), "prompt": p["prompt"],
               "response_a": p["response_a"], "response_b": p["response_b"],
               "label_a": p.get("label_a", "A"), "label_b": p.get("label_b", "B")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[done] {len(pairs)} pairs -> {args.out}")
