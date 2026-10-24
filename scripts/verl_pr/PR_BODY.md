## What

Add a single-GPU 24GB GRPO + LoRA example for Qwen2.5-Math-1.5B.

Official verl examples default to 8×80G. Users with a single 24GB card
(e.g. RTX 4090) hit OOM on rollout or actor update and have to guess
`n_gpus_per_node`, LoRA rank, vLLM `gpu_memory_utilization`, and
`max_response_length`. This script fixes a verified config.

## Changes

- `examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh`
- `examples/grpo_trainer/README.md`: add a "24GB single GPU" section

## Config

- Qwen2.5-Math-1.5B-Instruct, LoRA r=32 / alpha=32 / all-linear
- vLLM colocate, `gpu_memory_utilization=0.45`, TP=1
- `max_prompt_length=1024`, `max_response_length=3072`
- `train_batch_size=4`, `rollout.n=8`, micro-batch=1
- GRPO with KL loss (`kl_loss_coef=0.001`), `loss_agg_mode=token-mean`

## Test

- Hardware: RTX 4090 24GB
- Stack: torch 2.8.0+cu128, vllm 0.11.0, verl v0.8.0
- Command: `MODEL=... DATA=... STEPS=20 bash examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh`
- Result: 20-step smoke passes; logs show `step:0` / `step:1`; peak GPU ~18GB.

## Notes

This is intentionally an example/docs PR, not a core algorithm change.
