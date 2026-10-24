# RFT / expert iteration（P1）

目标：在单卡 24G 上做一个**正结果锚点**，同时接住候选人的模仿学习背景。

**问题**：在 1.5B + 单卡 + 有限 token 预算下，离线拒绝采样微调（RFT）和在线 GRPO 谁更划算？

**流程**：

```
base 模型对 MATH 7.4k 每题采 8 条
  → math_verify 判分，只留答对的
  → 选最短的正确回答（控制长度）
  → LoRA SFT（只对 response 算 loss）
  → 合并 adapter
  → 评测 MATH-500 / AIME24/25（3 seeds）
  → 对比 base / RFT / GRPO（R1、MATH R1）
```

## 1. 准备环境

```bash
source /root/autodl-tmp/venv-verl/bin/activate
cd /root/autodl-tmp/work/repo   # 本 repo
pip install math-verify
```

## 2. 一键运行

```bash
bash scripts/rft/run_rft.sh
```

默认路径：

- base：`/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct`
- 数据：`/root/autodl-tmp/work/data/math_train_verl/train.parquet`
- 输出：`/root/autodl-tmp/work/data/rft_math/`、`/root/autodl-tmp/work/runs/rft-lora`、`/root/autodl-tmp/work/runs/rft-merged`

## 3. 分步命令

```bash
# 1) 采样 + 过滤（约 2–4h，7.4k × 8 条）
python scripts/rft/sample_and_filter.py \
  --model /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --data /root/autodl-tmp/work/data/math_train_verl/train.parquet \
  --out /root/autodl-tmp/work/data/rft_math/train.jsonl \
  --n 8 --temperature 0.7 --top-p 0.95 --max-tokens 2048

# 2) LoRA SFT（约 1–2h）
python scripts/rft/train_sft_lora.py \
  --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --data /root/autodl-tmp/work/data/rft_math/train.jsonl \
  --out /root/autodl-tmp/work/runs/rft-lora \
  --epochs 1 --lr 1e-4 --max-len 2048

# 3) 合并（约 2 分钟）
python scripts/rft/merge_lora.py \
  --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/runs/rft-lora \
  --out /root/autodl-tmp/work/runs/rft-merged

# 4) 评测（MATH-500 n=4 ×3 seed；AIME n=8 ×3 seed）
python scripts/data/prepare_eval_sets.py --out /root/autodl-tmp/work/data/eval
python scripts/eval/eval_math.py --model /root/autodl-tmp/work/runs/rft-merged \
  --data /root/autodl-tmp/work/data/eval/math500.jsonl \
  --out /root/autodl-tmp/work/results/rft_math500.json --n 4 --seeds 42,43,44
```

## 4. 验收

- [ ] `sample_and_filter_stats.json` 里 `kept > 3000`（或至少覆盖大部分 MATH）；
- [ ] SFT loss 下降（0.5 以下）；
- [ ] RFT 的 MATH-500 相对 base **+2pp 以上**（如果没达到，见 §6）；
- [ ] 3 seed mean±std 已算；
- [ ] 同时报 pass@k 和平均长度。

## 5. 对比组

| 组 | 数据 | 配置 |
|---|---|---|
| base | — | 不训练 |
| RFT | 过滤后的正确样本 | LoRA SFT 1 epoch |
| GRPO | MATH 7.4k / DAPO 5k | 已有 R1、MATH R1（见手册 §6.10） |

**同 token 预算**：RFT 的 SFT token 数 vs GRPO 的 rollout token 数要记录，不能只比分数。

## 6. 如果 RFT 也不涨

这不是失败，是一个更完整的结论：

> "在 1.5B + 单卡 + 有限预算下，离线 RFT 和在线 GRPO 都没有带来可测的 MATH-500 提升；说明瓶颈不是 on/off-policy，而是模型容量/数据/评测噪声。"

面试讲法：先给数字，再说边界，再说"如果给 7B/更多 token 会怎么做"。

## 7. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 采样生成 OOM | vLLM 占比太高 | `--gpu-mem 0.8`；降 `--max-model-len` |
| `kept` 很少 | base 在 MATH 上 pass@8 低 | 增大 n 或换更容易的数据；不要硬跑 |
| SFT loss 不降 | lr 太小 / labels 错 | 先过拟合 50 条；检查 prompt mask |
| 合并后评测和 base 一样 | adapter 没 merge | 用 `merge_lora.py`，不要直接 `model_merger` |
| 评测全 0 | 答案解析失败 | 先对 gold 自检；检查 `\boxed{}` |
