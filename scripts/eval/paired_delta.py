#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired delta vs base with bootstrap CI over questions.

Compares two eval_math.py result jsons on the same question set (same data file,
same seeds). For each seed it pairs per-question scores, then bootstraps the
question-level paired delta (2000 resamples) to get a 95% CI.

Usage:
  python scripts/eval/paired_delta.py --base results/p1-base-math500.json \
    --run results/p1-r1-math500.json [--subset aime24_2024_ids.json] [--out delta.md]
"""
import argparse
import json
import random
import statistics


def load(path):
    return json.load(open(path))


def paired(base, run, subset=None):
    """Return (delta_pp, ci_low, ci_high, per_seed_deltas, n_questions)."""
    base_by_seed = {b["seed"]: {p["id"]: p for p in b["problems"]} for b in base["problems"]}
    run_by_seed = {b["seed"]: {p["id"]: p for p in b["problems"]} for b in run["problems"]}
    seeds = sorted(set(base_by_seed) & set(run_by_seed))
    per_seed = []
    for s in seeds:
        bmap, rmap = base_by_seed[s], run_by_seed[s]
        ids = sorted(set(bmap) & set(rmap))
        if subset:
            ids = [i for i in ids if i in subset]
        diffs = [rmap[i]["avg"] - bmap[i]["avg"] for i in ids]
        per_seed.append(sum(diffs) / len(diffs))
    # bootstrap over questions, pooling seeds
    rng = random.Random(0)
    all_diffs = []
    for s in seeds:
        bmap, rmap = base_by_seed[s], run_by_seed[s]
        ids = sorted(set(bmap) & set(rmap))
        if subset:
            ids = [i for i in ids if i in subset]
        all_diffs.append([rmap[i]["avg"] - bmap[i]["avg"] for i in ids])
    n = len(all_diffs[0])
    boots = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        vals = [statistics.mean(d[i] for i in idx) for d in all_diffs]
        boots.append(statistics.mean(vals))
    boots.sort()
    delta = statistics.mean(per_seed)
    return delta * 100, boots[50] * 100, boots[1949] * 100, [d * 100 for d in per_seed], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--labels", default="")
    args = ap.parse_args()
    subset = set(json.load(open(args.subset))["ids"]) if args.subset else None
    base = load(args.base)
    labels = args.labels.split(",") if args.labels else [r.split("/")[-1] for r in args.run]
    lines = ["| run | Δ avg (pp) | 95% CI (question bootstrap) | per-seed Δ | n |",
             "|---|---:|---|---:|---:|"]
    for label, path in zip(labels, args.run):
        d, lo, hi, per, n = paired(base, load(path), subset)
        lines.append(f"| {label} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{', '.join(f'{x:+.2f}' for x in per)} | {n} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
