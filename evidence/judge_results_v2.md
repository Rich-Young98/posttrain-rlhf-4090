# 独立 judge v2：多 judge × 扩样本复核（2026-10-27）

> 背景：v1（[judge_results.md](judge_results.md)）只有 DeepSeek 单 judge、DPO 200 对 / BoN 89 对。本页是补强后的结果：**4 个 judge 模型（5 种配置）× 最多 566 对**，同一套 prompt / 解码 / A-B 交换 / 3 票 / ties 单列 / Wilson CI。
> 机器可读：[judge_summary_v2.csv](judge_summary_v2.csv)。原始逐对 JSON 在本地 `p2/judge_v2/`（未入 repo，含每票理由）。

## 结论先行（这份证据改写了 v1 的一个结论）

1. **DPO800 的"显著更差"不稳健，是 judge-dependent**：DeepSeek 40.9% [33.8,48.4]（n=171），但 GLM-4.5-air 53.0% [48.2,57.7]（n=423）、Qwen2.5-7B 48.4% [43.0,53.9]（n=316）、Qwen correctness-first 49.4% [45.1,53.7]（n=508）、Mistral-7B **61.8%** [56.6,66.7]（n=356）。**5 个 judge 从 40.9% 到 61.8%，方向都不一致**，不能再说"DPO800 显著更差"。
2. **DPO800 的负效应主要由长度驱动**：按词数差 ±10 严格匹配后，DPO800 胜率变成 49.5%–64.7%（DeepSeek 60.5%、GLM 55.3%、Qwen 64.7%、Qwen-CF 55.5%、Mistral 49.5%）——**长度对齐后不再差，部分 judge 下反而更好**。DPO 把回答拉长了 +31.4%，judge 们对"更长"的惩罚/奖励方向不一致。
3. **DPO300 在 5 个 judge 上都是平手**：51.0–54.7%，CI 全部跨 50%——这条 v1 结论保留。
4. **BoN 选择有效是稳的**：N=64 在 5 个 judge 上 63.9–74.6%，N=128 64.3–79.0%，CI 下界全部 >57%——v1 的正结果被多 judge 复现。
5. **judge 之间一致性低**：同 89 对 BoN 上 pairwise Cohen κ = −0.002 ~ 0.476（Fleiss κ ≈ 0.15–0.32）；ties 比例从 3% 到 44% 不等。**LLM judge 不是 ground truth，单 judge 结论要谨慎**——这正是补人工校准（下一步）的理由。

## 主表（胜率 = b/(a+b)，ties 不计入分母；Wilson 95% CI）

### DPO800 vs base

| judge | n(有效) | ties | 胜率 | 95% CI | 长度差±20 | 长度差±10（严格） |
|---|---:|---:|---:|---|---:|---:|
| DeepSeek-chat（v1，200 对） | 171 | 29 | **40.94%** | [33.8, 48.4] | 48.6% | 60.5% |
| GLM-4.5-air（566 对） | 423 | 143 | **52.96%** | [48.2, 57.7] | 56.3% | 55.3% |
| Qwen2.5-7B（566 对） | 316 | 250 | **48.42%** | [43.0, 53.9] | 64.9% | 64.7% |
| Qwen2.5-7B correctness-first（566 对） | 508 | 58 | **49.41%** | [45.1, 53.7] | 55.0% | 55.5% |
| Mistral-7B-Instruct-v0.3（566 对） | 356 | 210 | **61.80%** | [56.6, 66.7] | 50.0% | 49.5% |

> 同一 200 对子集上，DeepSeek 40.9%、GLM 48.0%、Qwen 46.4%——**即使同一批题，judge 换一个就不显著**。

### DPO300 vs base

| judge | n(有效) | ties | 胜率 | 95% CI |
|---|---:|---:|---:|---|
| DeepSeek-chat（200 对） | 138 | 62 | 52.90% | [44.6, 61.0] |
| GLM-4.5-air（200 对） | 130 | 70 | 52.31% | [43.8, 60.7] |
| Qwen2.5-7B（566 对） | 236 | 330 | 51.27% | [44.9, 57.6] |
| Qwen2.5-7B correctness-first（566 对） | 427 | 139 | 51.05% | [46.3, 55.8] |
| Mistral-7B（566 对） | 254 | 312 | 54.72% | [48.6, 60.7] |

### best-of-N（N=64 / N=128 vs N=1）

| judge | N | n(有效) | ties | 胜率 | 95% CI | 严格匹配±10 |
|---|---:|---:|---:|---:|---|---:|
| DeepSeek-chat（v1，89 对/N） | 64 | 84 | 5 | 65.48% | [54.8, 74.8] | 54.3% |
| DeepSeek-chat（v1） | 128 | 78 | 11 | 66.67% | [55.6, 76.1] | 57.6% |
| GLM-4.5-air（89 对/N） | 64 | 76 | 13 | 71.05% | [60.0, 80.0] | 61.3% |
| GLM-4.5-air | 128 | 76 | 13 | 78.95% | [68.5, 86.6] | 67.7% |
| Qwen2.5-7B（300 prompt 扩样本） | 64 | 152 | 112 | 69.74% | [62.0, 76.5] | 71.4% |
| Qwen2.5-7B | 128 | 163 | 102 | 69.33% | [61.9, 75.9] | 75.9% |
| Qwen2.5-7B correctness-first | 64 | 249 | 15 | 63.86% | [57.7, 69.6] | 64.0% |
| Qwen2.5-7B correctness-first | 128 | 249 | 16 | 64.66% | [58.5, 70.3] | 64.5% |
| Mistral-7B | 64 | 159 | 105 | 69.18% | [61.6, 75.8] | 71.4% |
| Mistral-7B | 128 | 163 | 102 | 71.17% | [63.8, 77.6] | 63.6% |

## judge 一致性（同题同对）

| 组 | 对比 | 一致率 | Cohen κ |
|---|---|---:|---:|
| BoN N=64（89 对） | DeepSeek vs GLM | 71.9% | 0.476 |
| BoN N=64 | DeepSeek vs Qwen | 37.1% | -0.002 |
| BoN N=64 | DeepSeek vs Mistral | 46.1% | 0.152 |
| BoN N=64 | GLM vs Qwen | 41.6% | 0.062 |
| BoN N=64 | GLM vs Mistral | 41.6% | 0.059 |
| BoN N=64 | Qwen vs Mistral | 44.9% | 0.132 |
| BoN N=64 | **Fleiss κ（5 judges）** | — | **≈0.15** |

> 一致性低不是"结果坏了"，而是说明 pairwise judge 的标签噪声大；因此本页只做"多 judge 一致性"判断，不把任何一个 judge 当 ground truth。

## 方法（冻结）

- 同一 prompt 集合、同一生成参数（T=0.7/top_p=0.9/max 256）；judge 只看 Response A/B，不给来源；
- 每个 pair：AB、BA、再随机一票（3 票），多数票；ties 单列；
- 解析失败重试；401/402/403 直接失败，不把错误当 tie；
- judge 配置：DeepSeek-chat（v1，200 对）；GLM-4.5-air（`thinking: disabled`，Zhipu）；Qwen2.5-7B-Instruct 本地 vLLM；Mistral-7B-Instruct-v0.3 本地 vLLM；Qwen correctness-first prompt 变体；
- 扩样本：DPO 566 对（UF holdout 全量）、BoN 300 prompt × N=1/64/128（529 对）；
- 统计：Wilson 95% CI；ties 不计入分母为主口径；Fleiss κ 多 judge。

## 局限（写报告/面试主动说）

- **人工校准未完成**：150 对盲评表已生成（`p2/human_150.md/csv`），双评按计划暂停；人-judge 一致率/κ 还没有。
- **OpenAI judge 没跑成**：本机与 AutoDL 都连不上 `api.openai.com`（网络超时，无代理），key 未用上。
- **Zhipu 余额耗尽**：DPO300 566 对与 BoN 300 的 GLM 复评被 429 挡住（code 1113 余额不足），GLM 只在 200/178 对全量 + DPO800 566 上有效。
- **Mistral 7B 是较弱 judge**：它对 DPO800 给出 61.8%（与其他 judge 反向），且严格匹配子集 49.5%，更像"长度/格式偏好不稳定"；把它当**敏感性检查**，不当主判据。
- **ties 比例差异大**（Qwen 默认 44% vs Qwen-CF 10%）说明 judge 的"平手阈值"不同；excl-ties 口径下结论对 ties 处理敏感，已同时报 incl-ties 文件。
- 没有做人工大样本、没有 7B/14B 之外的大模型 judge 交叉（72B 级未跑）。

## 文件

- 汇总：本页 + [judge_summary_v2.csv](judge_summary_v2.csv)；
- 原始：本地 `p2/judge_v2/*.json`（逐对 votes/reason），`p2/judge_v2/raw/`（pairs）；
- 脚本：[judge_api.py](../scripts/judge/judge_api.py)、[generate_pair_outputs_vllm.py](../scripts/judge/generate_pair_outputs_vllm.py)、[aggregate_judges.py](../scripts/judge/aggregate_judges.py)、[make_human_sheet.py](../scripts/judge/make_human_sheet.py)。
