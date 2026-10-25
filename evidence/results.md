# 结果汇总（证据链）

> 完整机器可读表：[run_index.csv](run_index.csv)；项目一逐 run：[p1_results.md](p1_results.md)；独立 judge 汇总：[judge_results.md](judge_results.md) / [judge_summary.csv](judge_summary.csv)；RFT：[rft_results.md](rft_results.md)。本页给面试前一页速览。

## 项目一（RLVR / GRPO，Qwen2.5-Math-1.5B + LoRA）

| run | 数据 | 步数 | AIME24 avg/pass | AIME25 avg/pass | MATH-500 avg/pass | train acc |
|---|---|---:|---|---|---|---:|
| base | — | 0 | 7.50 / 10.00 | 9.17 / 20.00 | 72.15 / 82.00 | — |
| R1 | DAPO 5k | 300 | 10.83 / 13.33 | 7.50 / 16.67 | 71.35 / 83.00 | 28.28% |
| R1b | DAPO 5k | 300 (lr 2e-5) | 5.00 / 10.00 | 10.00 / 20.00 | 71.65 / 83.40 | 27.99% |
| R1c | DAPO 5k | 600 | 未评测 | 未评测 | 未评测 | 29.43% |
| MATH-R1 | MATH 7.4k | 300 | 3.33 / 6.67 | 5.00 / 16.67 | 72.65 / 80.40 | 74.20% |

**读法**：

- R1 的 AIME24 看似 +3.3pp，但 AIME25 -1.7pp、MATH-500 -0.8pp，R1b 直接掉到 5%；没有一致提升。
- MATH-R1 把 train acc 从 30% 拉到 74%，但 eval 不涨、AIME 掉约 4pp。
- 结论：1.5B + LoRA + 300–600 步的学习信号/预算不足；这是受控负结果，不是"调参没调对"。

## 项目一补充：RFT / expert iteration（负结果）

> 详情 [rft_results.md](rft_results.md)。MATH 7.4k 采样 8 条、过滤保留 5,062（68.3%）、最短正确回答 SFT 1 epoch。

| 指标 | base | RFT | Δ |
|---|---:|---:|---:|
| MATH-500 avg@4 | 58.08% | 57.98% | **-0.10pp** |
| MATH-500 pass@4 | 66.47% | 65.47% | **-1.00pp** |
| AIME24 avg@8 | 9.49% | 9.77% | **+0.28pp** |
| AIME25 avg@8 | 9.17% | 7.64% | **-1.53pp** |

**读法**：变化都在 seed/题目噪声内，RFT 没有可测提升。与 GRPO 合起来：**on-policy 和 offline 都不是瓶颈，容量/数据/预算才是。**

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
