# verl PR / 上游贡献

> 2026-10-28：项目一/二冻结，单卡 24G 的 GRPO 示例已提给 verl 上游。

## PR #8405（已提交）

- 链接：https://github.com/verl-project/verl/pull/8405
- 标题：`examples: add single-GPU 24GB GRPO + LoRA recipe (Qwen2.5-1.5B)`
- 内容：
  - `examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh`（新增）
  - `examples/grpo_trainer/README.md` 增加 “Single GPU (24 GB) + LoRA” 段
- 配置：Qwen2.5-Math-1.5B + LoRA r=32/all-linear + vLLM colocate
  (`gpu_memory_utilization=0.45`) + `max_response_length=3072` + micro-batch=1；
  显式 `use_shm=False` / `rollout.load_format=dummy` / `rollout.layered_summon=False`。
- 验证：RTX 4090 24G、torch 2.8.0+cu128、vllm 0.11.0、verl v0.8.0；
  20 步 smoke + 600 步训练（峰值 ~20GB）。
- 状态：open（等 maintainer review；可能要求 rebase 到 v0.9.x 配置名）。

## PR2：variance-weighted sampler（未提交）

`scripts/verl_pr/variance_sampler.py` + `test_variance_sampler.py` 是**候选设计**，不是已接进
verl 的补丁：

- 思路：GRPO 梯度 ∝ 组内奖励方差；全对/全错组 advantage=0。DAPO dynamic sampling 是“在线丢弃”，
  这个 sampler 是“离线估计 pass-rate + 主动加权/课程”。
- 为什么没提：需要改 `create_dataloader`、加 config、跑单测和真实数据验证；当前仓库里只有
  独立实现和权重单测，没有接进 trainer，不适合直接开 PR。
- 结论：作为“设计/面试可讲”的候选保留，若之后继续做再提 feature PR。

## 复现 PR 里的验证

见仓库 `scripts/p1/run_grpo.sh`（与 PR 脚本同配置，带 manifest/日志/seed 处理）和
`p1/evidence/`、`autodl_archive_20261027/` 中的原始日志。
