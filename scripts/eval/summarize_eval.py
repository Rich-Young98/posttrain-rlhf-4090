#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize unified eval results into one markdown table (dual AIME labels).

Reads the per-run json files produced by eval_math.py (v2), optionally filters
AIME22-24 down to the AIME2024 (30-question) subset, and prints:
  run | MATH-500 avg@4/pass@4 | AIME22-24 avg@8/pass@8 | AIME2024(30) avg@8/pass@8 | AIME25 avg@8/pass@8

Usage:
  python scripts/eval/summarize_eval.py --dir /path/to/results/unified/main \
    --subset /path/to/aime24_2024_ids.json
"""
import argparse
import glob
import json
import os
import statistics


def agg(values):
    m = statistics.mean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def subset_summary(res, ids):
    """Recompute avg@n and pass@n on a question-id subset from per-seed problems."""
    rows = []
    for block in res["problems"]:
        probs = [p for p in block["problems"] if p["id"] in ids]
        if not probs:
            continue
        rows.append({
            "avg": sum(p["avg"] for p in probs) / len(probs),
            "pass": sum(1 for p in probs if max(p["scores"]) > 0.5) / len(probs),
        })
    return agg([r["avg"] for r in rows]), agg([r["pass"] for r in rows]), len(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ids = set()
    if args.subset and os.path.exists(args.subset):
        ids = set(json.load(open(args.subset))["ids"])

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        res = json.load(open(path))
        bench = res.get("benchmark") or ("math500" if "math500" in name else ("aime25" if "aime25" in name else "aime24"))
        runs.setdefault(name, {})[bench] = res

    lines = []
    header = ("| run | prompt | MATH-500 avg@4 | MATH-500 pass@4 | AIME22-24(90) avg@8 | "
              "AIME22-24(90) pass@8 | AIME2024(30 子集) avg@8 | AIME25(30) avg@8 | parse_fail | trunc |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines += [header, sep]
    detail = []
    for name, benches in sorted(runs.items()):
        cells = [name]
        mode = ""
        def fmt(bench, key, pct=True):
            r = benches.get(bench)
            if not r:
                return "—"
            a = r["aggregate"][key]
            return f"{a['mean']*100:.2f}±{a['std']*100:.2f}"
        m5 = benches.get("math500")
        if m5:
            mode = m5.get("prompt_mode", "")
        a90 = benches.get("aime24")
        a25 = benches.get("aime25")
        sub_avg = sub_pass = None
        if a90 and ids:
            (sa, ss), (pa, ps), n = subset_summary(a90, ids)
            sub_avg, sub_pass = (sa, ss), (pa, ps)
        cells.append(mode)
        cells.append(fmt("math500", "avg@n"))
        cells.append(fmt("math500", "pass@4"))
        cells.append(fmt("aime24", "avg@n"))
        cells.append(fmt("aime24", "pass@8"))
        cells.append(f"{sub_avg[0]*100:.2f}±{sub_avg[1]*100:.2f}" if sub_avg else "—")
        cells.append(fmt("aime25", "avg@n"))
        pf = m5 or a90
        cells.append(f"{pf['aggregate']['parse_fail_rate']['mean']*100:.2f}%" if pf else "—")
        cells.append(f"{pf['aggregate']['truncation_rate']['mean']*100:.2f}%" if pf else "—")
        lines.append("| " + " | ".join(cells) + " |")
        detail.append((name, benches, sub_avg, sub_pass))

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(text + "\n")
        # also dump a machine-readable json next to the markdown
        out_json = args.out[:-3] + ".json" if args.out.endswith(".md") else args.out + ".json"
        payload = {}
        for name, benches, sub_avg, sub_pass in detail:
            payload[name] = {
                "prompt_mode": (benches.get("math500") or {}).get("prompt_mode"),
                "math500": benches.get("math500", {}).get("aggregate"),
                "aime22_24": benches.get("aime24", {}).get("aggregate"),
                "aime2024_subset": ({"avg@n": {"mean": sub_avg[0], "std": sub_avg[1]},
                                     "pass@8": {"mean": sub_pass[0], "std": sub_pass[1]}}
                                    if sub_avg else None),
                "aime25": benches.get("aime25", {}).get("aggregate"),
            }
        json.dump(payload, open(out_json, "w"), indent=1, ensure_ascii=False)
        print(f"[done] -> {args.out} / {out_json}")


if __name__ == "__main__":
    main()
