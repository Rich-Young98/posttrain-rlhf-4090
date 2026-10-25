#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze judge JSON: overall / per-length-bin / per-label (N=64 vs N=128).

Why: judge_pairwise.py's summary uses pairs[0].label_b for the whole file and
does not report per-bin wins_a/ties. For best-of-N, N=64 and N=128 pairs are in
one file, so the summary label can be misleading. This script recomputes from
raw pairs, groups by label_b, and adds bootstrap 95% CI.

Usage:
  python scripts/judge/analyze_judge.py \
    --judge_json /root/autodl-tmp/work/results/dpo800_judge.json \
    --out /root/autodl-tmp/work/results/dpo800_analysis.json
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def wlen(s):
    return len((s or "").split())


def bin_of(a, b):
    diff = wlen(b) - wlen(a)
    if diff > 20:
        return "b_longer"
    if diff < -20:
        return "a_longer"
    return "similar"


def boot_ci(outcomes, n_boot=2000, seed=0):
    """outcomes: 1 for B win, 0 for A win (ties already removed)."""
    if not outcomes:
        return None, None
    rng = random.Random(seed)
    n = len(outcomes)
    vals = []
    for _ in range(n_boot):
        c = sum(outcomes[rng.randrange(n)] for _ in range(n))
        vals.append(c / n)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def summarize(pairs, label_a, label_b):
    wins_a = sum(p["final_winner"] == "A" for p in pairs)
    wins_b = sum(p["final_winner"] == "B" for p in pairs)
    ties = sum(p["final_winner"] == "tie" for p in pairs)
    errors = sum(
        p.get("final_winner") == "error"
        or any(str(v.get("reason", "")).upper().startswith("ERROR")
               for v in p.get("votes", []))
        for p in pairs
    )
    valid = len(pairs) - errors
    decisive = wins_a + wins_b
    wr = wins_b / decisive if decisive else None
    lo, hi = boot_ci([1] * wins_b + [0] * wins_a)
    bins = defaultdict(lambda: {"n": 0, "wins_a": 0, "wins_b": 0, "ties": 0})
    for p in pairs:
        k = bin_of(p["response_a"], p["response_b"])
        bins[k]["n"] += 1
        if p["final_winner"] == "A":
            bins[k]["wins_a"] += 1
        elif p["final_winner"] == "B":
            bins[k]["wins_b"] += 1
        elif p["final_winner"] == "tie":
            bins[k]["ties"] += 1
    for k, v in bins.items():
        d = v["wins_a"] + v["wins_b"]
        v["win_rate_b_excl_ties"] = round(v["wins_b"] / d, 4) if d else None
    return {
        "label_a": label_a, "label_b": label_b, "n": len(pairs),
        "valid": valid, "errors": errors,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "win_rate_b_excl_ties": round(wr, 4) if wr is not None else None,
        "win_rate_b_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "win_rate_b_incl_ties": round((wins_b + 0.5 * ties) / valid, 4) if valid else None,
        "length_bins": dict(bins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = json.load(open(args.judge_json, encoding="utf-8"))
    pairs = d["pairs"]
    labels = Counter(p.get("label_b", "B") for p in pairs)
    print("[labels]", dict(labels))
    report = {"file": args.judge_json, "groups": {}}
    if len(labels) == 1:
        report["groups"]["all"] = summarize(pairs, pairs[0].get("label_a", "A"),
                                            pairs[0].get("label_b", "B"))
    else:
        for lab in labels:
            sub = [p for p in pairs if p.get("label_b") == lab]
            report["groups"][lab] = summarize(sub, sub[0].get("label_a", "A"), lab)
    for k, v in report["groups"].items():
        print(f"\n===== {k} =====")
        print(json.dumps(v, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
