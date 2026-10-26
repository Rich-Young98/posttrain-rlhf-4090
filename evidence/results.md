# 结果汇总（证据链）

> 完整机器可读表：[run_index.csv](run_index.csv)；项目一逐 run：[p1_results.md](p1_results.md)；独立 judge 汇总：[judge_results.md](judge_results.md) / [judge_summary.csv](judge_summary.csv)；RFT：[rft_results.md](rft_results.md)。本页给面试前一页速览。

## 项目一（RLVR / GRPO，Qwen2.5-Math-1.5B + LoRA）

> **主口径 = P1-UNIFIED 统一复评**（2026-10-26）：同一评测器 / 同题 / 同解码 / 同 parser / 3 个 eval seed。旧 P1-LEGACY 数字只作历史，不与本表混比（见文末）。

| run | 数据 | 步数 | MATH-500 avg@4 | AIME22–24(90) avg@8 | AIME2024(30 子集) avg@8 | AIME25(30) avg@8 | train acc |
|---|---|---:|---:|---:|---:|---:|---:|
| base | — | 0 | **75.42±0.29** | **9.35±0.29** | 10.56±0.86 | **10.56±0.52** | — |
| R1 | DAPO 5k | 300 | 74.68±0.39 | 9.12±0.24 | 10.83±0.68 | 9.44±0.39 | 28.28% |
| R1b | DAPO 5k | 300 (lr 2e-5) | 75.12±0.13 | 9.81±0.57 | 11.25±1.48 | 9.44±0.39 | 27.99% |
| R1c | DAPO 5k | 600 | 未评测 | 未评测 | 未评测 | 未评测 | 29.43% |
| MATH-R1 | MATH 7.4k | 300 | 75.38±0.57 | 9.17±0.41 | 11.67±0.68 | 10.42±1.23 | 74.20% |
| RFT | MATH 7.4k → 5,062 | 317 SFT | **75.47±0.08** | 9.35±0.35 | 10.42±0.59 | **6.94±0.39** | — |

**配对 delta（vs base，MATH-500 / AIME22–24 / AIME25）**：R1 −0.73[−1.33,−0.10] / −0.23 / −1.11；R1b −0.30 / +0.46 / −1.11；MATH-R1 −0.03 / −0.19 / −0.14；RFT +0.05 / 0.00 / **−3.61[−7.08,−0.56]**。

**读法**：

- **没有任何方法带来稳定提升**：MATH-500 全部在 base 的 −0.73 ~ +0.05pp；除 R1 的 −0.73pp（CI 不含 0，即"略差"）外都跨 0。
- **旧结论"R1b AIME 崩到 5.00""MATH-R1 AIME 掉 4pp"在统一协议下消失**：R1b = 9.81 / 11.25，MATH-R1 = 9.17 / 11.67，都在 base 噪声内 → 旧值是单 seed 30 题噪声。
- **RFT 的负效应只在 AIME25**：−3.61pp（3 seed 全负）；MATH-500 +0.05pp、AIME22–24 0.00pp → 离线自蒸馏没有泛化增益。
- 结论：在已测试的 1.5B + LoRA + 300–600 步设置下没有跨评测集一致提升；**不能据此断言主因是容量/数据/预算**（单次训练、R1c 未评测）。
- 细节与逐题 jsonl：父目录 `p1/evidence/eval_unified/`。

**旧 P1-LEGACY 数字（仅历史，不可与上表混比）**：base MATH-500 72.15 / AIME2024 7.50；R1 71.35 / 10.83；R1b 71.65 / 5.00；MATH-R1 72.65 / 3.33；RFT 新协议 58.08→57.98（另一套评测）。旧"新协议 base 58.08"已确认是判分 bug（模板只占 0.5pp，主因是 boxed 正则截断 + 数字兜底）。

## 项目二（RLHF / RM + DPO + best-of-N）

### RM

| 指标 | 数值 |
|---|---:|
| RewardBench 500 | 80.8% |
| RewardBench 全量 2,985 | **79.23%** |
| UF holdout 300 | 95.0% |
| length-only baseline | 35.58% |
| chosen 更短桶 | **56.61%**（n=454） |
| chosen 更长桶 | **99.13%**（n=231） |
| 长度匹配子集 | 87.63% |
| Pearson(len diff, gap) | **0.4325** |

**读法**：不是纯长度（length-only 35.6%，长度匹配 87.6%），但有强长度先验；79.2% 里有一部分是长度红利。

### DPO（RM proxy vs 独立 judge）

| 指标 | DPO 300 | DPO 800 |
|---|---:|---:|
| RM proxy win | 62.0% | **67.5%** |
| 独立 judge win | **52.9%** | **40.9%** |
| 95% CI | 44.6–61.0% | 33.8–48.4% |
| ties | 62 | 29 |
| similar 桶（judge） | 51.85% | 48.61% |
| DPO 更长时（judge） | 45.95% | **32.18%** |
| 平均长度 | +5.5% | **+31.4%** |
| 人工 20 条 | 未做 | 4–5 更好 / **8 更差** / 8 差不多 |

**读法**：DPO300 与 base 平手；DPO800 **显著更差**。RM proxy 高估 **26.6pp**，是长度/风格 shortcut。DPO 是负结果，不是"对齐成功"。

### best-of-N

| N | proxy mean | gold mean | 平均长度 |
|---:|---:|---:|---:|
| 1 | -1.943 | -2.276 | 143.6 |
| 8 | +0.379 | -0.141 | 153.0 |
| 64 | +1.453 | +0.942 | 159.4 |
| 128 | **+1.808** | **+1.256** | **161.9** |

**读法**：N=1..128 proxy/gold 同向上升，长度 +12.7%，长度相同 pair 的 gold 仍 +2.97；**未观察到经典过优化拐点**（negative result）。但同一批选中回答在独立 judge 下 **N=64 65.5%、N=128 66.7%**，similar 桶约 66%——**“选择”本身是正结果**。held-out gold RM 与 proxy 同偏置，不能当独立 judge。

## 一句话

> 同一套 1.5B/24G 预算：项目一 GRPO + RFT 都是受控负结果；项目二 RM 有长度偏置、DPO 被独立 judge 证伪（40.9%）、best-of-N 选择有效（65.5/66.7%）。**选择 vs 优化，是两种不同的 reward-hacking 风险。**

## 证据索引

- 每个 run 的数字 → [run_index.csv](run_index.csv)
- judge 方法/CI/长度桶 → [judge_results.md](judge_results.md)
- RFT → [rft_results.md](rft_results.md)
- DPO 生成评测原始汇总 → [dpo_eval_200.json](dpo_eval_200.json)
- best-of-N 曲线原始汇总 → [bon_summary.json](bon_summary.json)
- 人工复核 20 条 → [dpo_manual_review_20.md](dpo_manual_review_20.md)
- RM 全量分析原始输出 → [rm_full_analysis.txt](rm_full_analysis.txt)
