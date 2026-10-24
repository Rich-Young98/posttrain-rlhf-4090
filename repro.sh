#!/usr/bin/env bash
# Minimal reproduction entrypoint. See README.md for environment.
set -euo pipefail
PY=${PY:-/root/autodl-tmp/venv-verl/bin/python}
WORK=${WORK:-/root/autodl-tmp/work}

echo "== P0: independent judge =="
$PY scripts/judge/generate_pair_outputs.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --adapter "$WORK/runs/dpo-b005/final" \
  --prompts_parquet "$WORK/data/uf_rm/holdout.parquet" \
  --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
  --out "$WORK/results/dpo800_pairs.jsonl"
$PY scripts/judge/judge_pairwise.py \
  --pairs "$WORK/results/dpo800_pairs.jsonl" \
  --out "$WORK/results/dpo800_judge.json" --backend api --votes 3

echo "== P1: RFT =="
bash scripts/rft/run_rft.sh
