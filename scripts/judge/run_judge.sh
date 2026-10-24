#!/usr/bin/env bash
# One-shot: DPO 300/800 + best-of-N independent judge.
# Requires: JUDGE_API_KEY (or OPENAI_API_KEY), JUDGE_BASE_URL, JUDGE_MODEL.
set -euo pipefail
WORK=${WORK:-/root/autodl-tmp/work}
BASE=${BASE:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}
PY=${PY:-python}

mkdir -p "$WORK/results"

echo "== 1/4 DPO 800 pairs =="
$PY scripts/judge/generate_pair_outputs.py \
  --base_model "$BASE" --adapter "$WORK/runs/dpo-b005/final" \
  --prompts_parquet "$WORK/data/uf_rm/holdout.parquet" \
  --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
  --out "$WORK/results/dpo800_pairs.jsonl" --label_a base --label_b dpo800

echo "== 2/4 DPO 300 pairs =="
$PY scripts/judge/generate_pair_outputs.py \
  --base_model "$BASE" --adapter "$WORK/runs/dpo-b005-300/final" \
  --prompts_parquet "$WORK/data/uf_rm/holdout.parquet" \
  --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
  --out "$WORK/results/dpo300_pairs.jsonl" --label_a base --label_b dpo300

echo "== 3/4 judge DPO =="
$PY scripts/judge/judge_pairwise.py --pairs "$WORK/results/dpo800_pairs.jsonl" \
  --out "$WORK/results/dpo800_judge.json" --backend api --votes 3
$PY scripts/judge/judge_pairwise.py --pairs "$WORK/results/dpo300_pairs.jsonl" \
  --out "$WORK/results/dpo300_judge.json" --backend api --votes 3

echo "== 4/4 judge best-of-N =="
SEL=${BON_SEL:-"$WORK/results/bon128_selected .jsonl"}
if [ ! -f "$SEL" ]; then
  SEL=$(find "$WORK" -name "bon128_selected*" 2>/dev/null | head -1 || true)
fi
if [ -n "$SEL" ] && [ -f "$SEL" ]; then
  $PY scripts/judge/judge_bon_pairs.py --selected "$SEL" --ns 64 128 \
    --out "$WORK/results/bon_pairs.jsonl"
  $PY scripts/judge/judge_pairwise.py --pairs "$WORK/results/bon_pairs.jsonl" \
    --out "$WORK/results/bon_judge.json" --backend api --votes 3
else
  echo "[skip] $SEL not found; copy bon128_selected.jsonl first"
fi
echo "[done] see $WORK/results/*judge.json"
