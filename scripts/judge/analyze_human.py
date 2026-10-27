#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze blind human annotations against multi-judge results.

Inputs:
  --annotated  filled csv: code, winner(A/B/tie), optional notes/scores
  --mapping    code,id,experiment,order produced by the sheet generator
  --judge-dir  p2/judge_v2 (uses the 566-pair judge JSONs per experiment)

Outputs:
  - human win rate per experiment (adapter side), Wilson CI
  - human-judge agreement + Cohen's kappa per judge
  - strict length-matched (+/-10 words) human win rate
  - markdown report (and JSON if --json-out)
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

JUDGE_FILES = {
    "dpo800": [("deepseek", "deepseek_dpo800_566.json"), ("glm-4.5-air", "zhipu_dpo800_566.json"),
               ("qwen7b", "qwen_dpo800_566.json"), ("qwen7b-cf", "qwen_cf_dpo800_566.json"),
               ("mistral7b", "mistral_dpo800_566.json")],
    "dpo300": [("deepseek", "deepseek_dpo300_566.json"), ("glm-4.5-air", "zhipu_dpo300_566.json"),
               ("qwen7b", "qwen_dpo300_566.json"), ("qwen7b-cf", "qwen_cf_dpo300_566.json"),
               ("mistral7b", "mistral_dpo300_566.json")],
    "N=64": [("deepseek", "deepseek_bon_300.json"), ("glm-4.5-air", "zhipu_bon_300.json"),
             ("qwen7b", "qwen_bon_300.json"), ("qwen7b-cf", "qwen_cf_bon_300.json"),
             ("mistral7b", "mistral_bon_300.json")],
    "N=128": [("deepseek", "deepseek_bon_300.json"), ("glm-4.5-air", "zhipu_bon_300.json"),
              ("qwen7b", "qwen_bon_300.json"), ("qwen7b-cf", "qwen_cf_bon_300.json"),
              ("mistral7b", "mistral_bon_300.json")],
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cohen_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for p in pairs if p[0] == c) / n) *
             (sum(1 for p in pairs if p[1] == c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def norm(x):
    x = (x or "").strip().lower()
    if x in ("a", "b", "tie", "t"):
        return x.upper() if x in ("a", "b") else "tie"
    return None


def load_judges(judge_dir):
    out = {}
    for exp, files in JUDGE_FILES.items():
        for judge, fname in files:
            p = os.path.join(judge_dir, fname)
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            for r in d["pairs"]:
                if r.get("label_b") != exp or r.get("final_winner") not in ("A", "B", "tie"):
                    continue
                out[(exp, judge, r.get("id"))] = {
                    "winner": r["final_winner"],
                    "len_a": len((r.get("response_a") or "").split()),
                    "len_b": len((r.get("response_b") or "").split()),
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--judge-dir", default="p2/judge_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    mapping = {r["code"]: r for r in csv.DictReader(open(args.mapping, encoding="utf-8"))}
    ann = [r for r in csv.DictReader(open(args.annotated, encoding="utf-8")) if norm(r.get("winner(A/B/tie)", r.get("winner", "")))]
    judges = load_judges(args.judge_dir)

    rows = []
    for r in ann:
        code = r.get("code")
        m = mapping.get(code)
        if not m:
            continue
        h = norm(r.get("winner(A/B/tie)", r.get("winner", "")))
        if h is None:
            continue
        orig = {"A": "A", "B": "B"}.get(h) if m["order"] == "AB" else {"A": "B", "B": "A"}.get(h)
        orig = "tie" if h == "tie" else orig
        rows.append({"code": code, "id": m["id"], "experiment": m["experiment"],
                     "human": h, "human_orig": orig})

    lines = ["# 人工盲评结果\n"]
    summary = {}
    for exp in ["dpo800", "dpo300", "N=64", "N=128"]:
        er = [r for r in rows if r["experiment"] == exp]
        if not er:
            continue
        wins = sum(1 for r in er if r["human_orig"] == "B")
        loss = sum(1 for r in er if r["human_orig"] == "A")
        ties = len(er) - wins - loss
        lo, hi = wilson(wins, wins + loss)
        summary[exp] = {"n": len(er), "adapter_wins": wins, "base_wins": loss, "ties": ties,
                        "win_rate": wins / (wins + loss) if wins + loss else None,
                        "ci": [lo, hi]}
        lines.append(f"\n## {exp}\n")
        lines.append(f"- n={len(er)}，adapter 胜 {wins} / base 胜 {loss} / tie {ties}"
                     + (f"，胜率 **{wins/(wins+loss)*100:.1f}%**（Wilson 95% CI {lo*100:.1f}–{hi*100:.1f}%）" if wins + loss else ""))
        # length-matched strict
        strict = []
        for r in er:
            for (e, j, i), v in judges.items():
                if e == exp and i == r["id"]:
                    if abs(v["len_b"] - v["len_a"]) <= 10:
                        strict.append(r)
                    break
        if strict:
            sw = sum(1 for r in strict if r["human_orig"] == "B")
            sl = sum(1 for r in strict if r["human_orig"] == "A")
            lines.append(f"- 严格长度匹配（±10 词）n={sw+sl}，adapter 胜率 "
                         f"{sw/(sw+sl)*100:.1f}%" if sw + sl else "- 严格长度匹配样本不足")
        # agreement per judge
        lines.append("\n| judge | 可比对 n | 一致率 | Cohen κ | 该 judge 胜率 |")
        lines.append("|---|---:|---:|---:|---:|")
        for judge in ["deepseek", "glm-4.5-air", "qwen7b", "qwen7b-cf", "mistral7b"]:
            pairs = []
            jw = jl = 0
            for r in er:
                v = judges.get((exp, judge, r["id"]))
                if not v:
                    continue
                pairs.append((r["human_orig"], v["winner"]))
                if v["winner"] == "B":
                    jw += 1
                elif v["winner"] == "A":
                    jl += 1
            if not pairs:
                continue
            agree = sum(1 for a, b in pairs if a == b) / len(pairs)
            k = cohen_kappa(pairs)
            lines.append(f"| {judge} | {len(pairs)} | {agree*100:.1f}% | "
                         f"{k:.3f} | {jw/(jw+jl)*100:.1f}% |" if jw + jl else
                         f"| {judge} | {len(pairs)} | {agree*100:.1f}% | {k:.3f} | — |")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
