#!/usr/bin/env bash
# Judge best-of-N only (N=1 vs N=64/128). No DPO re-generation.
set -euo pipefail
WORK=${WORK:-/root/autodl-tmp/work}
PY=${PY:-python}
SEL=${BON_SEL:-"$WORK/results/bon128_selected .jsonl"}
if [ ! -f "$SEL" ]; then
  echo "[search] BON_SEL not found, searching $WORK ..."
  SEL=$(find "$WORK" -iname "*bon128*selected*" 2>/dev/null | head -1 || true)
fi
if [ -z "$SEL" ] || [ ! -f "$SEL" ]; then
  echo "[error] cannot find bon128_selected*.jsonl under $WORK"
  echo "run: find /root/autodl-tmp/work -iname '*bon128*selected*'"
  echo "then: BON_SEL=/path/to/file bash scripts/judge/run_judge_bon.sh"
  exit 1
fi
echo "[data] $SEL"
$PY scripts/judge/judge_bon_pairs.py --selected "$SEL" --ns 64 128 \
  --out "$WORK/results/bon_pairs.jsonl"
$PY scripts/judge/judge_pairwise.py --pairs "$WORK/results/bon_pairs.jsonl" \
  --out "$WORK/results/bon_judge.json" --backend api --votes 3
