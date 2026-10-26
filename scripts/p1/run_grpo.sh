#!/usr/bin/env bash
# Project 1: GRPO on a single RTX 4090 24G.
# Stack: verl v0.8.0 + vLLM 0.11.0 + FSDP + LoRA (rank 32).
#
# History:
#   - original runs were launched on AutoDL; this wrapper was reconstructed from the
#     resolved config printed in d4_r1_300.log / d4_r1b_lr2e5.log / d4_r1c_600.log /
#     p1_math_r1_300.log
#   - 2026-10-26: fixed the 4 known mismatches with the real runs (use_shm,
#     rollout.load_format, rollout.layered_summon, and the non-existent trainer.seed),
#     and made SAVE_FREQ / DATA_SEED explicit so a long run can keep step300 + step600.
#
# Usage:
#   MODE=dapo STEPS=300 LR=1e-5 EXP=p1-r1-300 bash scripts/p1/run_grpo.sh
#   MODE=dapo STEPS=600 SAVE_FREQ=300 DATA_SEED=42 EXP=p1-r1c-rerun-600 \
#     bash scripts/p1/run_grpo.sh
set -euo pipefail

MODE=${MODE:-dapo}                 # dapo | math
STEPS=${STEPS:-300}
LR=${LR:-1e-5}
DATA_SEED=${DATA_SEED:-42}         # data.seed + actor data_loader_seed + fsdp seed
SAVE_FREQ=${SAVE_FREQ:-$STEPS}
KEEP_CKPT=${KEEP_CKPT:-2}
EXP=${EXP:-p1-${MODE}-${STEPS}}

PY=${PY:-/root/autodl-tmp/venv-verl/bin/python}
VERL=${VERL:-/root/autodl-tmp/work/code/verl}
WORK=${WORK:-/root/autodl-tmp/work}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct}
GPU_MEM=${GPU_MEM:-0.45}

case "$MODE" in
  dapo) TRAIN="$WORK/data/dapo_5k_verl/train.parquet" ;;
  math) TRAIN="$WORK/data/math_train_verl/train.parquet" ;;
  *) echo "MODE must be dapo or math, got: $MODE" >&2; exit 2 ;;
esac

mkdir -p "$WORK/logs" "$WORK/runs"
LOG="$WORK/logs/${EXP}.log"
MANIFEST="$WORK/logs/${EXP}_manifest.txt"

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "exp: $EXP  steps=$STEPS  save_freq=$SAVE_FREQ  data_seed=$DATA_SEED"
  echo "verl_commit: $(git -C "$VERL" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "verl_status:"; git -C "$VERL" status --short 2>/dev/null || true
  echo "model_sha256: $(sha256sum "$MODEL/model.safetensors" | cut -d' ' -f1)"
  echo "model_config_sha256: $(sha256sum "$MODEL/config.json" | cut -d' ' -f1)"
  echo "data_sha256: $(sha256sum "$TRAIN" | cut -d' ' -f1)"
  echo "patched_files:"
  sha256sum "$VERL/verl/utils/reward_score/__init__.py" \
            "$VERL/verl/utils/attention_utils.py" \
            "$VERL/verl/utils/model.py" \
            "$VERL/verl/workers/config/model.py" 2>/dev/null || true
  echo "versions:"
  "$PY" -m pip list 2>/dev/null | grep -iE '^(torch|vllm|verl|transformers|peft|ray|math-verify|datasets) ' || true
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  df -h "$WORK" | tail -1
} > "$MANIFEST" 2>&1

echo "[p1] mode=$MODE steps=$STEPS lr=$LR data_seed=$DATA_SEED save_freq=$SAVE_FREQ exp=$EXP"
echo "[p1] data=$TRAIN"
echo "[p1] log=$LOG  manifest=$MANIFEST"

cd "$VERL"
PYTHONUNBUFFERED=1 "$PY" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN" \
  data.val_files="$TRAIN" \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.seed="$DATA_SEED" \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.data_loader_seed="$DATA_SEED" \
  actor_rollout_ref.actor.fsdp_config.seed="$DATA_SEED" \
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
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$KEEP_CKPT" \
  trainer.test_freq=-1 \
  trainer.project_name=rlvr-interview \
  trainer.experiment_name="$EXP" \
  2>&1 | tee "$LOG"

echo "[p1] done. checkpoint under $VERL/checkpoints/rlvr-interview/$EXP"
echo "[p1] next: merge each global_step adapter with scripts/p1/merge_lora_peft.py, then eval with scripts/eval/eval_math.py"
