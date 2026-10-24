# verl PR 预案（framework delta）

目标：让简历上出现"给 verl 提交过 PR"的可验证信号；同时把 P1 的实验变成框架能力。

## PR 1（先做，最容易 merged）：单卡 24G GRPO + LoRA 示例

**标题**：`examples: add single-GPU 24GB GRPO + LoRA recipe (Qwen2.5-1.5B)`

**改什么**：
- `examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh`（本目录脚本）
- `examples/grpo_trainer/README.md` 增加一段"24GB single GPU"说明
- 可选：`docs/start/quickstart.rst` 的 OOM 段落链到该脚本

**为什么**：官方示例默认 8×80G，24G 单卡用户要自己摸 `n_gpus_per_node=1`、
LoRA rank、`gpu_memory_utilization`、`max_response_length`；这个脚本把已验证配置固化。

**测试**：AutoDL 4090 24G，verl v0.8.0 + vllm 0.11.0 + torch 2.8.0；
跑 20 步 smoke，日志出现 `step:0/step:1`，显存 < 20G。

**PR 描述模板**：
> This PR adds a single-GPU 24GB GRPO + LoRA example for Qwen2.5-1.5B.
> Official examples assume 8 GPUs; 24GB users often hit OOM on rollout/actor.
> The script uses LoRA r=32, vLLM colocate (`gpu_memory_utilization=0.45`),
> max_response_length=3072, micro-batch=1. Verified on RTX 4090 24G
> (torch 2.8.0, vllm 0.11.0, verl v0.8.0): 20-step smoke passes, peak ~18GB.

## PR 2（P1 之后）：variance-weighted prompt sampler

**标题**：`[feat] add variance-weighted prompt sampler for GRPO`

**改什么**：
- 新增 `verl/trainer/ppo/sampler.py`（本目录 `variance_sampler.py`）
- `data.sampler.type=uniform|variance_weighted`、`pass_rate_path`、`p_low`、`p_high`
- `create_dataloader` 里按配置替换 sampler
- 单测：`tests/trainer/ppo/test_variance_sampler.py`（本目录测试）
- 文档：说明 pass-rate 需离线估计（base model n=8/16）

**为什么**：GRPO 的梯度 ∝ 组内奖励方差；全对/全错组 advantage=0。
DAPO 的 dynamic sampling 是"在线丢弃"，这个 sampler 是"离线估计 + 主动加权/课程"，
可以在同一 token 预算下优先采样有学习信号的题。

**测试**：
- 单测：权重归一化、全 0 退化、band 过滤；
- 端到端：MATH 7.4k，uniform vs variance_weighted，同 token 预算；
- 指标：MATH-500 avg@4、pass@8、零方差组比例、有效样本率。

**注意**：feature PR 可能被要求改设计/加 benchmark；先提 PR 1 拿 merged，
再提 PR 2（open + 清晰 diff 也能讲）。

## 操作步骤

```bash
# 1) fork verl 到自己的 GitHub，clone
git clone https://github.com/<you>/verl.git && cd verl
git checkout -b feat/single-gpu-24g-grpo

# 2) 放文件
cp <repo>/scripts/verl_pr/run_qwen2_5_1p5b_lora_24g.sh examples/grpo_trainer/
# 编辑 examples/grpo_trainer/README.md 加一节

# 3) 本地验证
MODEL=... DATA=... STEPS=20 bash examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh

# 4) 提交
git add ... && git commit -m "examples: add single-GPU 24GB GRPO + LoRA recipe"
git push -u origin feat/single-gpu-24g-grpo

# 5) 开 PR（网页或 gh）
gh pr create --title "examples: add single-GPU 24GB GRPO + LoRA recipe" \
  --body-file <repo>/scripts/verl_pr/PR_BODY.md
```

## 面试怎么说

- "PR #___（merged/open）：给 verl 加了单卡 24G GRPO+LoRA 示例；官方示例默认 8 卡，24G 用户会 OOM。"
- "另一个 sampler PR 是 open 状态：diff 在 ___；实验上我在 MATH 7.4k 上对比了 uniform vs variance-weighted，指标是 ___。"
- **不要说**"我改了 verl 核心"——除非 diff 真的是核心；说"示例/配置 + sampler 模块"。
