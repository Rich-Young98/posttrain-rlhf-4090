#!/usr/bin/env bash
# RFT (rejection-sampling fine-tuning) pipeline.
# Run from the repo root: bash scripts/rft/run_rft.sh
set -euo pipefail

PY=${PY:-/root/autodl-tmp/venv-verl/bin/python}
BASE=${BASE:-/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct}
DATA=${DATA:-/root/autodl-tmp/work/data/math_train_verl/train.parquet}
WORK=${WORK:-/root/autodl-tmp/work}
N=${N:-8}
LIMIT=${LIMIT:-0}

mkdir -p "$WORK/data/rft_math" "$WORK/runs" "$WORK/results"

echo "== 1/4 sample + filter =="
$PY scripts/rft/sample_and_filter.py \
  --model "$BASE" --data "$DATA" --limit "$LIMIT" \
  --out "$WORK/data/rft_math/train.jsonl" \
  --n "$N" --temperature 0.7 --top_p 0.95 --max-tokens 2048

echo "== 2/4 LoRA SFT =="
$PY scripts/rft/train_sft_lora.py \
  --base "$BASE" --data "$WORK/data/rft_math/train.jsonl" \
  --out "$WORK/runs/rft-lora" --epochs 1 --lr 1e-4 --max-len 2048

echo "== 3/4 merge =="
$PY scripts/rft/merge_lora.py \
  --base "$BASE" --adapter "$WORK/runs/rft-lora" --out "$WORK/runs/rft-merged"

echo "== 4/4 eval (base vs RFT) =="
$PY scripts/data/prepare_eval_sets.py --out "$WORK/data/eval" --datasets math500,aime24,aime25
for MODEL in "$BASE" "$WORK/runs/rft-merged"; do
  NAME=$(basename "$MODEL")
  $PY scripts/eval/eval_math.py --model "$MODEL" --data "$WORK/data/eval/math500.jsonl" \
    --out "$WORK/results/${NAME}_math500.json" --n 4 --seeds 42,43,44 --max-new-tokens 2048
  $PY scripts/eval/eval_math.py --model "$MODEL" --data "$WORK/data/eval/aime24.jsonl" \
    --out "$WORK/results/${NAME}_aime24.json" --n 8 --seeds 42,43,44 --max-new-tokens 2048
done
echo "[done] see $WORK/results/"
