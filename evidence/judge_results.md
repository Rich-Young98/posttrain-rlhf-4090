> ⚠️ **本页是 v1（单 judge，DeepSeek）历史记录**。2026-10-27 已补 5 judge × 566 对复核：[judge_results_v2.md](judge_results_v2.md)。
> v1 的“DPO800 显著差于 base”**已被 v2 降级为 judge-dependent + 长度驱动**（GLM 53.0%、Qwen 48.4%、Mistral 61.8%；长度匹配 ±10 后 49.5–64.7%）。下图/v1 数字仅作历史对照。

# 独立 judge 结果（2026-10-25，DeepSeek pairwise）

> 口径：DPO n=200；BoN N=64 / N=128 各 89 对。A/B 位置交换 + 3 票多数；胜率 = b / (a + b)，**ties 不计入分母**；CI 为 Wilson 95%。
> 机器可读：[judge_summary.csv](judge_summary.csv)。
> 原始逐 pair JSON（`dpo300_judge.json`、`dpo800_judge.json`、`bon_judge.json`）产生在 AutoDL `/root/autodl-tmp/work/results/`，实例关机后未拉回本地；本 repo 保留**逐组汇总**，数字与报告 §11、实验日志 D15/D16 一致。

## 方法

- judge：DeepSeek `deepseek-chat`（与 1.5B RM 不同来源）；
- 输入：同一 prompt + 两条回答，不告诉 judge 哪条来自哪个模型；
- 顺序：AB / BA 各问一次，再补一票（共 3 票），多数票为 `final_winner`；
- 解析：只接受 A/B/tie；解析失败重试；401/402/403 直接失败退出，不把错误当 tie；
- 统计：ties 单独报；`win_rate = b/(a+b)`；Wilson 95% CI；按长度差拆 `a_longer` / `similar` / `b_longer`。

## DPO 300 vs base

| 组 | n | a | b | ties | 胜率（不含 ties） | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| overall | 200 | 65 | 73 | 62 | **52.90%** | 44.60–61.04% |
| similar | 133 | 39 | 42 | 52 | 51.85% | — |
| b_longer | 46 | 20 | 17 | 9 | 45.95% | — |
| a_longer | 21 | 6 | 14 | 1 | 70.00% | — |

**结论**：CI 跨 50%，DPO300 与 base 无显著差异；similar 桶 51.85%，没有质量提升。RM proxy 的 62.0% 被高估。

## DPO 800 vs base

| 组 | n | a | b | ties | 胜率（不含 ties） | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| overall | 200 | 101 | 70 | 29 | **40.94%** | 33.84–48.43% |
| similar | 82 | 37 | 35 | 10 | 48.61% | — |
| b_longer | 105 | 59 | 28 | 18 | **32.18%** | — |
| a_longer | 13 | 5 | 7 | 1 | 58.33% | — |

**结论**：DPO800 **显著差于 base**；当 DPO 更长时只赢 32.18%。RM proxy 67.5% → 独立 judge 40.94%，**高估 26.6pp**。这是长度/风格 shortcut，不是对齐。

## best-of-N（N=1 为基线）

| 组 | n | a | b | ties | 胜率（不含 ties） | 95% CI | similar 桶 |
|---|---:|---:|---:|---:|---:|---|---:|
| N=64 | 89 | 29 | 55 | 5 | **65.48%** | 54.83–74.77% | 66.04% |
| N=128 | 89 | 26 | 52 | 11 | **66.67%** | 55.64–76.13% | 65.96% |

**结论**：N=64 与 N=128 都显著优于 N=1；similar 桶仍有 ~66%，说明提升不是单纯变长；N=128 没有下降，**N≤128 未观察到过优化**。这与“proxy 选得更好”一致，但与“DPO 优化更好”不是一回事。

## 一句话

> 同一个 RM：**用来做 best-of-N 选择**能提升独立 judge 胜率（65.5/66.7%）；**用来做 DPO 优化**反而把模型带偏（40.9%）。选择 vs 优化，是两种完全不同的风险。

## 局限（面试主动说）

- 单 judge、单次评测；judge 本身也可能有位置/长度/风格偏好，只是比同分布 gold RM 独立；
- DPO 200 对、BoN 各 89 对；DPO300 的 CI 约 ±8pp，BoN 的 CI 约 ±12pp；
- 未做人工大样本校准，也没有第二个 judge 交叉验证；
- 原始逐 pair JSON 未入库（实例关机），但逐组 a/b/ties 与 CI 均在本页和 `judge_summary.csv`。
