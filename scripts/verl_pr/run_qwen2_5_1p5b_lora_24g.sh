#!/usr/bin/env bash
# Single-GPU GRPO + LoRA example for a 24GB card (e.g., RTX 4090).
# Model: Qwen2.5-Math-1.5B-Instruct; data: any verl-format math parquet.
#
# Tested stack (2026-10): torch 2.8.0+cu128, vllm 0.11.0, verl v0.8.0.
# Usage:
#   MODEL=/path/Qwen2.5-Math-1.5B-Instruct \
#   DATA=/path/math/train.parquet \
#   bash run_qwen2_5_1p5b_lora_24g.sh
set -euo pipefail
MODEL=${MODEL:?set MODEL}
DATA=${DATA:?set DATA}
SAVE=${SAVE:-./checkpoints/single_gpu_grpo_lora}
STEPS=${STEPS:-300}
GEN=${GEN:-8}
BS=${BS:-4}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_RESP=${MAX_RESP:-3072}
GPU_MEM=${GPU_MEM:-0.45}

python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA" \
  data.val_files="$DATA" \
  data.train_batch_size="$BS" \
  data.max_prompt_length="$MAX_PROMPT" \
  data.max_response_length="$MAX_RESP" \
  data.gen_batch_size="$GEN" \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_shm=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$GEN" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BS" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  algorithm.adv_estimator=grpo \
  trainer.logger='["console"]' \
  trainer.project_name=single-gpu-grpo \
  trainer.experiment_name=qwen2_5_1p5b_lora_24g \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.default_local_dir="$SAVE"
