# 项目一结果：单卡 GRPO + RFT（受控负结果）

> 一键汇总：[results.md](results.md)；run 级 CSV：[run_index.csv](run_index.csv)；面试口径：[../docs/项目一_结果与面试口径.md](../docs/项目一_结果与面试口径.md)；完整报告在父目录 `项目一_最终报告.md`。
> 原始日志在父目录（`d4_r1_300.log` 等）；本页是面试前一页速览 + 证据索引。

## 1. 训练配置（共同项）

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-Math-1.5B-Instruct + LoRA（r=32，alpha=32，all-linear） |
| 框架 | verl v0.8.0 + vLLM 0.11.0 colocate + FSDP，单卡 4090 24G |
| GRPO | `adv_estimator=grpo`，`rollout.n=8`，`train_batch_size=4`，KL loss 0.001，`token-mean` |
| 长度 | prompt 1024 / response 3072，`gpu_memory_utilization=0.45`，seed 42 |
| 评测 | AIME24/25 `avg@4 / pass@4`；MATH-500 `avg@4 / pass@4`；训练集 reward 即 train acc（DAPO 为 ±1，需换算；MATH 为 0/1） |

## 2. 主表

| run | 数据 | 步数 | lr | train acc | AIME24 avg/pass | AIME25 avg/pass | MATH-500 avg/pass | 耗时 | 结论 |
|---|---|---:|---:|---:|---|---|---|---:|---|
| base | — | 0 | — | — | 7.50 / 10.00 | 9.17 / 20.00 | 72.15 / 82.00 | — | 基线 |
| R1 | DAPO 5k | 300 | 1e-5 | 28.28% | **10.83 / 13.33** | 7.50 / 16.67 | 71.35 / 83.00 | 5.94h | AIME24 +3.3pp 但 AIME25/MATH 反降；无一致提升 |
| R1b | DAPO 5k | 300 | 2e-5 | 27.99% | 5.00 / 10.00 | 10.00 / 20.00 | 71.65 / 83.40 | 5.55h | 加 lr 只增大 KL（2.3e-4），AIME24 崩到 5% |
| R1c | DAPO 5k | 600 | 1e-5 | 29.43% | 未评测 | 未评测 | 未评测 | 9.37h | 加步数仍卡 ~30% train acc，停止 |
| MATH-R1 | MATH 7.4k | 300 | 1e-5 | **74.20%** | 3.33 / 6.67 | 5.00 / 16.67 | 72.65 / 80.40 | 4.27h | train acc 高但 eval 不涨，AIME 掉约 4pp |

## 3. 逐 run 读法

### R1（DAPO 5k，300 步，lr 1e-5）

- train acc 28.28%，KL 1.65e-4，5.94h；
- AIME24 10.83 / 13.33（base 7.50 / 10.00）看起来 +3.3pp；
- 但 AIME25 7.50 / 16.67（base 9.17 / 20.00）掉 1.7pp，MATH-500 71.35 vs 72.15 掉 0.8pp；
- 单点、n=4，AIME 30 题的 95% CI 约 ±5pp → **不能叫提升**。

### R1b（只把 lr 改成 2e-5）

- train acc 27.99%，KL 2.30e-4（最大 9.24e-4）；
- AIME24 5.00 / 10.00，直接腰斩；AIME25 10.00 / 20.00 略高，但 MATH-500 71.65 仍低于 base；
- 说明“加大 lr”只让策略更新更猛，不产生可测收益。

### R1c（只把步数改成 600）

- train acc 29.43%；100 步窗口 25.9→27.5→31.0→31.4→31.0→29.8%，没有继续上升趋势；
- 没花 GPU 评测 → 写“未评测”，不补数字；
- 结论：加步数不解决 ~30% 平台。

### MATH-R1（换成 MATH 7.4k，300 步）

- train acc 74.20%（前 100 73.47%、后 100 75.84%、后 50 73.56%），长度 570，KL ~1e-4；
- 25/300 步全对、0 步全错：大量组内 advantage 为 0，学习信号反而少；
- MATH-500 72.65 / 80.40（base 72.15 / 82.00）：avg +0.5pp、pass −1.6pp；
- AIME24 3.33 / 6.67、AIME25 5.00 / 16.67：OOD 掉约 4pp；
- 结论：MATH 让训练“可学”，但没有转成泛化。

### RFT（离线 expert iteration，负结果）

- 数据：MATH 7,414 题 × 8 采样 → `math_verify` 过滤 → 保留 **5,062（68.28%）**；选最短正确回答；
- 训练：LoRA SFT 1 epoch，317 步，loss 0.046，19.5min；
- 评测：`eval_math.py`，3 个采样 seed，报 mean±std；
- 结果：MATH-500 avg@4 **58.08 → 57.98**（−0.10pp）、pass@4 66.47 → 65.47；AIME24 avg@8 9.49 → 9.77（+0.28）；AIME25 9.17 → 7.64（−1.53）；
- 结论：离线自蒸馏只在拟合已会做的题，没有新增学习信号；与 GRPO 合起来，**on-policy/off-policy 都不是瓶颈**。

> 注意：RFT 的 MATH-500 base 58.08 用的是新协议（`eval_math.py`，n=4，3 seed），项目一老 base 72.15 用的是 `d2_tools.py` 协议，**两个数不能横比**。

## 4. 结论

1. **没有一致提升**：R1 的 AIME24 是单点噪声；R1b/R1c/MATH-R1 都不支持“继续调 lr/步数/数据能涨”。
2. **两个受控负结果**：在线 GRPO 与离线 RFT 都没有可测收益。
3. **瓶颈判断**：1.5B + LoRA + 24G + 300–600 步下，更可能是容量/数据/预算，而不是 lr、步数或 on/off-policy。
4. **工程产出**：单卡 24G 的 GRPO 闭环、数据去重/去污染、reward 路由、LoRA 合并、评测噪声控制。

## 5. 证据

| 内容 | 文件 |
|---|---|
| R1/R1b/R1c 日志 | 父目录 `d4_r1_300.log`、`d4_r1b_lr2e5.log`、`d4_r1c_600.log` |
| MATH-R1 日志 | 父目录 `p1_math_r1_smoke.log`、`p1_math_r1_300.log` |
| RFT 日志 | 父目录 `rft.log`、`train_stats.json` |
| 曲线 | [p1_curves/](p1_curves/)（R1、R1 vs R1b、R1 家族、MATH-R1） |
| 启动脚本 | [../scripts/p1/](../scripts/p1/)（按日志重建） |
| RFT 代码 | [../scripts/rft/](../scripts/rft/) |
| 评测器 | [../scripts/eval/eval_math.py](../scripts/eval/eval_math.py) |
| 原始 eval JSON | 仍在 AutoDL `results/`（`base_eval.json`、`r1_300_eval_hf.json`、`r1b_300_eval.json`、`math_r1_300_eval.json`），未入库 |

## 6. 限制（主动说）

- 单 seed 训练；评测 AIME 的 n=4/8，30 题噪声大；
- R1c 未评测（只有 train acc），不补数字；
- 原始训练 launch 脚本/数据准备脚本在 AutoDL，未入库；repo 是重建版；
- MATH-500 两套评测协议（72.15 vs 58.08），不横比；
- 只到 1.5B，不宣称规模外推。
