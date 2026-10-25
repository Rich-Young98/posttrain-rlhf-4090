#!/usr/bin/env bash
# Project 1: GRPO on a single RTX 4090 24G.
# Stack: verl v0.8.0 + vLLM 0.11.0 + FSDP + LoRA (rank 32).
#
# This script is reconstructed from the actual run configs printed in:
#   d4_r1_300.log / d4_r1b_lr2e5.log / d4_r1c_600.log / p1_math_r1_300.log
# The original runs were launched on AutoDL; this wrapper keeps the same
# hyper-parameters and makes the two data regimes explicit.
#
# Usage:
#   MODE=dapo STEPS=300 LR=1e-5 EXP=p1-r1-300 bash scripts/p1/run_grpo.sh
#   MODE=dapo STEPS=300 LR=2e-5 EXP=p1-r1b-lr2e5 bash scripts/p1/run_grpo.sh
#   MODE=dapo STEPS=600 LR=1e-5 EXP=p1-r1c-600 bash scripts/p1/run_grpo.sh
#   MODE=math STEPS=300 LR=1e-5 EXP=p1-math-r1-300 bash scripts/p1/run_grpo.sh
set -euo pipefail

MODE=${MODE:-dapo}                       # dapo | math
STEPS=${STEPS:-300}
LR=${LR:-1e-5}
SEED=${SEED:-42}
EXP=${EXP:-p1-${MODE}-${STEPS}}

PY=${PY:-/root/autodl-tmp/venv-verl/bin/python}
VERL=${VERL:-/root/autodl-tmp/work/code/verl}
WORK=${WORK:-/root/autodl-tmp/work}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct}
GPU_MEM=${GPU_MEM:-0.45}

case "$MODE" in
  dapo)
    TRAIN="$WORK/data/dapo_5k_verl/train.parquet"
    ;;
  math)
    TRAIN="$WORK/data/math_train_verl/train.parquet"
    ;;
  *)
    echo "MODE must be dapo or math, got: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$WORK/logs" "$WORK/runs"
LOG="$WORK/logs/${EXP}.log"
echo "[p1] mode=$MODE steps=$STEPS lr=$LR seed=$SEED"
echo "[p1] data=$TRAIN"
echo "[p1] log=$LOG"

cd "$VERL"

PYTHONUNBUFFERED=1 "$PY" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN" \
  data.val_files="$TRAIN" \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_shm=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=grpo \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.seed="$SEED" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$STEPS" \
  trainer.test_freq=-1 \
  trainer.project_name=rlvr-interview \
  trainer.experiment_name="$EXP" \
  2>&1 | tee "$LOG"

echo "[p1] done. checkpoint under $VERL/checkpoints/rlvr-interview/$EXP"
echo "[p1] next: merge LoRA with scripts/p1/merge_lora_peft.py, then eval with scripts/eval/eval_math.py"
