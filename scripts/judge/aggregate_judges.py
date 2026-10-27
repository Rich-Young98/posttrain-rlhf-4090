#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate multi-judge pairwise results: per-experiment table, CIs, agreement.

Reads judge JSONs produced by judge_pairwise.py (DeepSeek) or judge_api.py
(Zhipu/local vLLM). Experiment is inferred from each pair's label_b
(e.g. dpo800 / dpo300 / N=64 / N=128); judge name from summary["model"] or filename.

Usage:
  python scripts/judge/aggregate_judges.py --dir p2/judge_v2 --out p2/judge_v2/summary.md
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def fleiss_kappa(ratings, categories=("A", "B", "tie")):
    """ratings: list of list-of-categories (one per judge) per item."""
    if not ratings or not ratings[0]:
        return None
    n_items, n_raters = len(ratings), len(ratings[0])
    if n_raters < 2:
        return None
    counts = []
    for row in ratings:
        c = Counter(row)
        counts.append([c.get(cat, 0) for cat in categories])
    p_j = [sum(row[j] for row in counts) / (n_items * n_raters) for j in range(len(categories))]
    P_e = sum(p * p for p in p_j)
    P_i = []
    for row in counts:
        s = sum(x * x for x in row)
        P_i.append((s - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(pairs):
    """pairs: list of (cat1, cat2)."""
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for p in pairs if p[0] == c) / n) * (sum(1 for p in pairs if p[1] == c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def word_len(s):
    return len((s or "").split())


def load_judges(directory, extra_files=()):
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    files += list(extra_files)
    judges = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "pairs" not in d or "summary" not in d:
            continue
        s = d["summary"]
        name = s.get("model") or os.path.basename(f).split("_")[0]
        if name in ("dpo300", "dpo800", "bon"):
            name = "deepseek"
        key = f"{name}|{os.path.basename(f)}"
        judges[key] = d
    return judges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="extra judge JSON paths (e.g. DeepSeek)")
    args = ap.parse_args()
    judges = load_judges(args.dir, args.extra)

    per = defaultdict(list)  # (exp, judge) -> records
    if len(judges) == 0:
        raise SystemExit("no judge JSON found")
    for key, d in judges.items():
        for rec in d["pairs"]:
            exp = rec.get("label_b") or d["summary"].get("label_b", "?")
            per[(exp, key)].append(rec)
    # if the same judge was run on the same experiment with several sample sizes,
    # keep the largest run (avoid duplicate rows)
    def valid_n(recs):
        return sum(1 for r in recs if r.get("final_winner") in ("A", "B", "tie"))
    for (exp, key), recs in list(per.items()):
        jname = key.split("|")[0]
        same = [(e, k) for (e, k) in per if e == exp and k.split("|")[0] == jname]
        if len(same) > 1:
            keep = max(same, key=lambda ek: valid_n(per[ek]))
            for ek in same:
                if ek != keep:
                    del per[ek]

    exps = sorted({e for e, _ in per})
    # split mixed bon file if label_b has N=... (already per-record above)
    lines = ["# 多 judge 汇总（自动生成）\n"]
    lines.append("> 口径：胜率 = b/(a+b)，ties 不计入分母；Wilson 95% CI；长度严格匹配 = |len_b - len_a| <= 10 词。\n")
    for exp in exps:
        lines.append(f"\n## {exp}\n")
        lines.append("| judge | n | a | b | ties | win_rate_b | 95% CI | similar(±20词) | strict(±10词) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
        for (e, judge), recs in sorted(per.items()):
            if e != exp:
                continue
            valid = [r for r in recs if r["final_winner"] in ("A", "B", "tie")]
            a = sum(r["final_winner"] == "A" for r in valid)
            b = sum(r["final_winner"] == "B" for r in valid)
            t = len(valid) - a - b
            wr = b / (a + b) if (a + b) else None
            lo, hi = wilson(b, a + b)
            sim = [(r["final_winner"], word_len(r["response_b"]) - word_len(r["response_a"])) for r in valid]
            sim_b = [w for w, d in sim if abs(d) <= 20]
            strict_b = [w for w, d in sim if abs(d) <= 10]
            def rate(wins):
                x = [w for w in wins if w in ("A", "B")]
                return (sum(1 for w in x if w == "B") / len(x) * 100) if x else None
            lines.append("| %s | %d | %d | %d | %d | %s | %s | %s | %s |" % (
                judge.split("|")[0].split("/")[-1], len(valid), a, b, t,
                f"{wr*100:.2f}%" if wr is not None else "—",
                f"[{lo*100:.1f}, {hi*100:.1f}]" if lo is not None else "—",
                f"{rate(sim_b):.1f}%" if rate(sim_b) is not None else "—",
                f"{rate(strict_b):.1f}%" if rate(strict_b) is not None else "—"))

    # agreement
    lines.append("\n## Judge 一致性（逐对）\n")
    by_exp = defaultdict(dict)
    for (exp, judge), recs in per.items():
        for r in recs:
            by_exp[exp].setdefault(r["id"], {})[judge.split("|")[0]] = r["final_winner"]
    for exp in exps:
        rows = by_exp[exp]
        names = sorted({n for d in rows.values() for n in d})
        names = [n for n in names if sum(1 for d in rows.values() if n in d) >= min(2, len(names))][:6]
        if len(names) < 2:
            continue
        common = [d for d in rows.values()
                  if all(n in d and d[n] in ("A", "B", "tie") for n in names)]
        if len(common) < 5:
            continue
        lines.append(f"\n**{exp}**（n={len(common)} 对，{len(names)} judges）\n")
        lines.append("| pair | agreement | Cohen κ |")
        lines.append("|---|---:|---:|")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs = [(d[names[i]], d[names[j]]) for d in common]
                agree = sum(1 for a, b in pairs if a == b) / len(pairs)
                lines.append(f"| {names[i]} vs {names[j]} | {agree*100:.1f}% | {cohen_kappa(pairs):.3f} |")
        ratings = [[d[n] for n in names] for d in common]
        lines.append(f"| Fleiss κ（{len(names)} judges） | — | {fleiss_kappa(ratings):.3f} |")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
