# 项目一结果：单卡 GRPO + RFT（统一复评后的受控负结果）

> 2026-10-26 更新：完成 **P1-UNIFIED 统一复评**（同一评测器 / 同题 / 同解码 / 同 parser / 3 个 eval seed）。
> 主口径 = §2 表；旧 P1-LEGACY 数字移到 §5 仅作历史存档，**不与主表混比**。
> 一键汇总：[results.md](results.md)；run 级 CSV：[run_index.csv](run_index.csv)；面试口径：[../docs/项目一_结果与面试口径.md](../docs/项目一_结果与面试口径.md)；完整报告在父目录 `项目一_最终报告.md`。

## 1. 训练配置（4 个 GRPO run 的共同项）

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-Math-1.5B-Instruct + LoRA（r=32，alpha=32，all-linear） |
| 框架 | verl v0.8.0 + vLLM 0.11.0 colocate + FSDP，单卡 4090 24G |
| GRPO | `adv_estimator=grpo`，`rollout.n=8`，`train_batch_size=4`，KL loss 0.001，`token-mean` |
| 长度 | prompt 1024 / response 3072，`gpu_memory_utilization=0.45`；**单次训练：`data.seed=None`（数据顺序未显式 seed），42 只是 `actor.data_loader_seed`/`fsdp_config.seed` 默认值** |
| 训练 acc | DAPO 为 ±1 reward（换算 `(r+1)/2`），MATH 为 0/1 |

## 2. 统一复评主表（P1-UNIFIED，mean±std over 3 eval seeds）

| 项 | 值 |
|---|---|
| 评测器 | `scripts/eval/eval_math.py` v2.0，sha256 `41f75d8d…`，`evaluator_version=p1-eval-v2.0-20261026` |
| prompt / 解码 | legacy 模板（MATH-500 要求 `\boxed{}`，AIME 要求 `Answer:`）；T=0.6 / top_p=0.95 / max_new_tokens 3072 |
| 采样 | MATH-500 n=4；AIME n=8；seeds **42,43,44** |
| 判分 | 平衡 `\boxed{}` → `Answer:` 行 → None；math_verify（verl 兼容配置）；**无数字兜底** |
| 数据 | MATH-500 500 题；**AIME22–24 90 题**；AIME25 30 题；AIME2024(30) 为 90 题子集 |

| run | MATH-500 avg@4 | MATH-500 pass@4 | AIME22–24(90) avg@8 | AIME22–24(90) pass@8 | AIME2024(30 子集) avg@8 | AIME25(30) avg@8 |
|---|---:|---:|---:|---:|---:|---:|
| base | **75.42±0.29** | 85.73±0.47 | **9.35±0.29** | 22.59±0.52 | 10.56±0.86 | **10.56±0.52** |
| R1（DAPO 5k, 300, lr 1e-5） | 74.68±0.39 | 85.20±0.16 | 9.12±0.24 | 21.11±2.40 | 10.83±0.68 | 9.44±0.39 |
| R1b（lr 2e-5） | 75.12±0.13 | 85.73±0.66 | 9.81±0.57 | 22.59±1.39 | 11.25±1.48 | 9.44±0.39 |
| MATH-R1（MATH 7.4k） | 75.38±0.57 | 85.93±0.93 | 9.17±0.41 | 20.37±0.52 | 11.67±0.68 | 10.42±1.23 |
| RFT（离线 expert iteration） | **75.47±0.08** | 85.07±0.68 | 9.35±0.35 | 19.63±1.89 | 10.42±0.59 | **6.94±0.39** |
| R1c（600 步） | 未评测 | 未评测 | 未评测 | 未评测 | 未评测 | 未评测 |

**配对 delta（vs base，逐题配对 + 2000 次 question bootstrap）**

| run | MATH-500 Δ avg@4 | AIME22–24(90) Δ avg@8 | AIME25 Δ avg@8 |
|---|---|---|---|
| R1 | **−0.73** [−1.33, −0.10] | −0.23 [−1.02, +0.46] | −1.11 [−4.17, +1.67] |
| R1b | −0.30 [−0.97, +0.37] | +0.46 [−0.28, +1.30] | −1.11 [−3.47, +0.83] |
| MATH-R1 | −0.03 [−0.62, +0.60] | −0.19 [−1.20, +0.74] | −0.14 [−1.81, +1.67] |
| RFT | +0.05 [−1.13, +1.27] | +0.00 [−1.20, +1.34] | **−3.61 [−7.08, −0.56]** |

> 详细 delta（含 AIME2024 30 题子集、每 seed 值）见父目录 `p1/evidence/eval_unified/*_delta.md`。

## 3. 结论（统一协议下）

1. **没有方法带来稳定提升**：MATH-500 全部落在 base 的 −0.73 ~ +0.05pp；除 R1 的 −0.73pp（CI 不含 0，即"略差"）外都跨 0。负结果成立且更干净。
2. **"R1b 的 AIME24 崩到 5.00" 是旧协议噪声**：统一后 R1b AIME22–24 = 9.81、AIME2024 子集 = 11.25，都在 base 噪声内。lr 2e-5 的影响只剩"KL 更大"，不能写成能力崩塌。
3. **"MATH-R1 的 AIME 掉约 4pp" 也基本是噪声**：统一后 9.17 / 11.67 / 10.42，CI 都跨 0；MATH-R1 真正的问题仍是 **MATH-500 不涨（−0.03pp）**。
4. **RFT 唯一方向一致的负效应在 AIME25**：−3.61pp（CI [−7.08, −0.56]，3 个 seed 全负）；MATH-500 与 AIME22–24 不变 → 离线自蒸馏没有泛化增益，在更难的小样本集上可能有害（30 题，谨慎表述）。
5. **旧"新协议 base 58.08"是判分 bug**：模板只贡献 −0.53pp（legacy 75.42 vs bare 74.88，CI 跨 0）；其余来自旧 `eval_math.py` 的 boxed 正则截断嵌套大括号 + 数字兜底 + 2048 tokens。v2 parser 下 base = 75.42，比旧 P1-LEGACY 72.15 高 3.27pp（判分更准）。

## 4. 逐 run 读法

- **R1**：MATH-500 −0.73pp（CI 不含 0）是唯一"方向明确"的差异，但幅度 <1pp，且 AIME 全在噪声内；不能说涨，也不能说"崩"。
- **R1b**：MATH-500 −0.30pp 跨 0；AIME 与 base 同量级。旧结论"加 lr 打崩 AIME"被统一复评推翻，能保留的只有"KL 从 1.65e-4 涨到 2.3e-4（最大 9.24e-4）"。
- **MATH-R1**：train acc 74.2% 是四个 GRPO run 里唯一"可学"的，但 MATH-500 −0.03pp、AIME 全跨 0；"可学 ≠ 泛化"这句话仍成立。
- **RFT**：MATH-500 +0.05pp、AIME22–24 0.00pp、AIME25 −3.61pp；训练侧 loss 0.046、317 步。**不能说 RFT 有提升**；AIME25 的负效应可以讲，但要标注 30 题小样本。
- **R1c**：未评测（checkpoint/adapter 均缺失），所有 benchmark 单元格只能写"未评测/产物缺失"。

## 5. 历史协议数字（P1-LEGACY / P1-NEW，仅存档，不与主表混比）

| run | 数据 | 步数 | lr | train acc | AIME2024(旧协议,30) avg/pass | AIME25 avg/pass | MATH-500 avg/pass | 耗时 |
|---|---|---:|---:|---:|---|---|---|---:|
| base | — | 0 | — | — | 7.50 / 10.00 | 9.17 / 20.00 | 72.15 / 82.00 | — |
| R1 | DAPO 5k | 300 | 1e-5 | 28.28% | 10.83 / 13.33 | 7.50 / 16.67 | 71.35 / 83.00 | 5.94h |
| R1b | DAPO 5k | 300 | 2e-5 | 27.99% | 5.00 / 10.00 | 10.00 / 20.00 | 71.65 / 83.40 | 5.55h |
| R1c | DAPO 5k | 600 | 1e-5 | 29.43% | 未评测 | 未评测 | 未评测 | 9.37h |
| MATH-R1 | MATH 7.4k | 300 | 1e-5 | 74.20% | 3.33 / 6.67 | 5.00 / 16.67 | 72.65 / 80.40 | 4.27h |
| RFT | 5,062 正确样本 | 317 SFT | 1e-4 | — | 9.49 / 19.63 → 9.77 / 19.26（avg@8/pass@8） | 9.17 / 23.33 → 7.64 / 24.44 | 58.08 / 66.47 → 57.98 / 65.47（avg@4/pass@4） | 0.33h |

> 旧协议：AIME2024 30 题、MATH-500 带 `\boxed` 指令、n=4 无显式 seed；RFT 行是另一个新协议（90 题 AIME、裸题面、2048 tok）。三者互不可比。

## 6. 证据

| 内容 | 文件 |
|---|---|
| 统一复评主表与 delta | 父目录 `p1/evidence/eval_unified/`（README、summary、4 个 delta、parser_spotcheck） |
| 逐题 JSONL | `p1/evidence/eval_unified/results/*.jsonl`（含 completion/parsed/token/truncated） |
| 训练日志 | 父目录 `d4_r1_300.log`、`d4_r1b_lr2e5.log`、`d4_r1c_600.log`、`p1_math_r1_300.log`、`rft.log` |
| 历史评测 json | 父目录 `p1/artifacts/raw_eval/` |
| 曲线 | [p1_curves/](p1_curves/) |
| 启动脚本 / RFT | [../scripts/p1/](../scripts/p1/)、[../scripts/rft/](../scripts/rft/) |
| 评测器 v2 | [../scripts/eval/eval_math.py](../scripts/eval/eval_math.py)、[summarize_eval.py](../scripts/eval/summarize_eval.py)、[paired_delta.py](../scripts/eval/paired_delta.py) |

## 7. 诊断审计（Day 4，2026-10-26）

> 证据包：父目录 `p1/evidence/audit/`（README + dedup/rollout/patch/trainlog JSON + 30 条退化样本复核）。

| 审计项 | 结果 |
|---|---|
| 训练日志 group 级指标 | **没有记录** → 训练时零方差组比例无法回溯（只能估计） |
| base rollout 零方差组（DAPO 150 题×8、MATH 60 题×8，T=1.0） | **DAPO 54%**（65 全错 / 16 全对）、**MATH 77%**（39 全对 / 7 全错）→ advantage 恒 0 的组过半 |
| reward 路由补丁 | 无补丁：DAPO reward **0%**（98% `[INVALID]`）；补丁激活率 **97%**，救回 **28.0%**（=train acc 28.28%） |
| 训练 reward vs 独立 oracle | 1200 条 rollout **0 分歧**（362 对 / 838 错）→ reward 定义没有系统性错判 |
| 截断 | DAPO 0.08%、MATH 0% → 不是长度/截断问题 |
| DAPO 去重链 | 1,791,700 → 17,917（index）→ 17,398（prompt）；**5k 训练集 = RNG(42).shuffle(clean)[:5000] 逐条命中** |
| 去污染 | 13-gram 529 候选 → **14 条确认（全 MATH-500）**；只用 13-gram 会误删 515 条 |
| 30 条退化样本 | 14 计算/代数失误、8 概念/公式误用、4 条件漏读、2 推理构造、1 题面解析、1 未化简；**无截断** |

**诊断**：最硬的证据是"有效梯度样本太少"（零方差组 54%/77% × batch=4），叠加 LoRA 容量与 300–600 步预算；但单次训练、无 group 级日志、R1c 未评测 → **不能说是已证明的唯一主因**。

## 8. 限制（主动说）

- **单次训练、数据顺序未显式 seed**（`data.seed=None`；`trainer.seed` 在 verl v0.8 不存在）；本复评是"评测 3 seed"，不是"训练 3 seed"。
- **R1c 未评测且产物缺失**（无 adapter/checkpoint）。
- AIME parse_fail 3.3–4.6%（多为 3072 token 截断，trunc 1.6–3.2%）→ AIME 绝对值 ±1–2pp 打折扣；RFT trunc 最高（3.24%）。
- 每套 AIME 30–90 题：30 题子集 1 题 = 3.33pp，CI 宽，不要把 +0.3~+1.1pp 说成提升。
- 只到 1.5B，不宣称规模外推；旧协议数字只作历史。
