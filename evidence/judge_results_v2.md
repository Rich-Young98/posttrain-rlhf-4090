# 独立 judge v2：多 judge × 扩样本复核（2026-10-27）

> 背景：v1（[judge_results.md](judge_results.md)）只有 DeepSeek 单 judge、DPO 200 对 / BoN 89 对。本页是补强后的结果：**5 个 judge 模型 × 最多 566 对**，同一套 prompt / 解码 / A-B 交换 / 3 票 / ties 单列 / Wilson CI。
> 机器可读：[judge_summary_v2.csv](judge_summary_v2.csv)。原始逐对 JSON 在本地 `p2/judge_v2/`（未入 repo，含每票理由）。

## 结论先行（这份证据改写了两条 v1 结论）

1. **DPO800 的“显著更差”是被 200 对样本 + DeepSeek 单 judge 放大的假象**：DeepSeek 在 200 对上 40.94% [33.8,48.4]，但扩到 566 对后是 **47.53% [42.9,52.2]**（不再显著）；GLM 52.96% [48.2,57.7]、Qwen 48.42% [43.0,53.9]、Qwen-CF 49.41% [45.1,53.7]、Mistral **61.80%** [56.6,66.7]。**5 个 judge 里 4 个 CI 跨 50%，1 个显著为正（Mistral），没有一个是显著为负。**
2. **DPO800 的负效应主要由长度驱动**：长度差 ±10 词严格匹配后，5 个 judge 为 **49.5–64.7%**（DeepSeek 50.4、GLM 55.3、Qwen 64.7、Qwen-CF 55.5、Mistral 49.5）——长度对齐后不再差，部分 judge 下更好。
3. **DPO300 在 5 个 judge 上都是平手**：51.05–54.72%，CI 全部跨 50。
4. **BoN 选择有效是稳的**：N=64 在 5 个 judge 上 63.78–70.24%，N=128 64.66–71.17%，**CI 下界全部 >56%**；严格长度匹配后 63.6–75.9%——v1 的正结果被多 judge 复现。
5. **judge 之间一致性低**：同题 pairwise Cohen κ 约 0.0–0.48、ties 比例 3–44% → LLM judge 标签噪声大；单 judge 的“显著”不可信，**结论必须是“多 judge 共识”而不是“挑一个 judge”**。
6. **多 seed（42/43/44，双 judge 到三 judge）**：Qwen 48.42/46.18/48.99、GLM 52.96/47.94/48.54、DeepSeek 47.53/44.14/48.06——9 个格子里 8 个 CI 跨 50，1 个（DeepSeek seed43）显著为负；**没有一致方向**，单 seed/单 judge 都不足以定论。

## 主表（胜率 = b/(a+b)，ties 不计入分母；Wilson 95% CI；均含扩样本）

### DPO800 vs base

| judge | n（有效） | ties | 胜率 | 95% CI | 长度差±20 | 长度差±10（严格） |
|---|---:|---:|---:|---|---:|---:|
| DeepSeek-chat（566 对） | 446 | 120 | **47.53%** | [42.9, 52.2] | — | 50.4% |
| GLM-4.5-air（566 对） | 423 | 143 | **52.96%** | [48.2, 57.7] | 56.3% | 55.3% |
| Qwen2.5-7B（566 对） | 316 | 250 | **48.42%** | [43.0, 53.9] | 64.9% | 64.7% |
| Qwen2.5-7B correctness-first（566 对） | 508 | 58 | **49.41%** | [45.1, 53.7] | 55.0% | 55.5% |
| Mistral-7B（566 对） | 356 | 210 | **61.80%** | [56.6, 66.7] | 50.0% | 49.5% |

> v1 的 DeepSeek 200 对 = 40.94% [33.8,48.4]（见 [judge_results.md](judge_results.md)）；同一 judge 扩样本后回到 47.53%，说明 v1 的“显著更差”同时受**小样本**和**judge 选择**影响。

### DPO300 vs base

| judge | n（有效） | ties | 胜率 | 95% CI |
|---|---:|---:|---:|---|
| DeepSeek-chat（566 对） | 366 | 200 | 53.55% | [48.4, 58.6] |
| GLM-4.5-air（566 对） | 344 | 222 | 52.62% | [47.3, 57.8] |
| Qwen2.5-7B（566 对） | 236 | 330 | 51.27% | [44.9, 57.6] |
| Qwen2.5-7B correctness-first（566 对） | 427 | 139 | 51.05% | [46.3, 55.8] |
| Mistral-7B（566 对） | 254 | 312 | 54.72% | [48.6, 60.7] |

### best-of-N（N=64 / N=128 vs N=1）

| judge | N | n（有效） | ties | 胜率 | 95% CI | 严格匹配±10 |
|---|---:|---:|---:|---:|---|---:|
| DeepSeek-chat（300 prompt） | 64 | 196 | 38 | 63.78% | [56.8, 70.2] | 67.5% |
| DeepSeek-chat | 128 | 164 | 32 | 65.85% | [58.3, 72.7] | 65.6% |
| GLM-4.5-air | 64 | 205 | 59 | 70.24% | [63.7, 76.1] | 71.1% |
| GLM-4.5-air | 128 | 209 | 56 | 69.38% | [62.8, 75.2] | 72.4% |
| Qwen2.5-7B | 64 | 152 | 112 | 69.74% | [62.0, 76.5] | 71.4% |
| Qwen2.5-7B | 128 | 163 | 102 | 69.33% | [61.9, 75.9] | 75.9% |
| Qwen2.5-7B correctness-first | 64 | 249 | 15 | 63.86% | [57.7, 69.6] | 64.0% |
| Qwen2.5-7B correctness-first | 128 | 249 | 16 | 64.66% | [58.5, 70.3] | 64.5% |
| Mistral-7B | 64 | 159 | 105 | 69.18% | [61.6, 75.8] | 71.4% |
| Mistral-7B | 128 | 163 | 102 | 71.17% | [63.8, 77.6] | 63.6% |

## 多 seed（DPO800，三 judge，566 对/seed）

| seed | Qwen | GLM | DeepSeek |
|---|---:|---:|---:|
| 42 | 48.42% [43.0,53.9] | 52.96% [48.2,57.7] | 47.53% [42.9,52.2] |
| 43 | 46.18% [40.5,52.0] | 47.94% [43.2,52.8] | **44.14% [39.5,48.8]** |
| 44 | 48.99% [43.3,54.7] | 48.54% [43.8,53.4] | 48.06% [43.4,52.7] |
| 均值±sd | 47.86±1.49 | 49.81±2.74 | 46.58±2.30 |

- **9 个格子中 8 个 CI 跨 50，1 个显著为负（DeepSeek seed43）**；没有一致方向；
- 结论：**“单 seed 结论不稳”被证实；DPO800 没有稳健的方向性效应**（既不能说显著变差，也不能说显著变好）。

## judge 一致性（同题同对，旧 200/89 子集）

| 组 | 对比 | 一致率 | Cohen κ |
|---|---|---:|---:|
| BoN N=64（89 对） | DeepSeek vs GLM | 71.9% | 0.476 |
| BoN N=64 | DeepSeek vs Qwen | 37.1% | -0.002 |
| BoN N=64 | DeepSeek vs Mistral | 46.1% | 0.152 |
| BoN N=64 | GLM vs Qwen | 41.6% | 0.062 |
| BoN N=64 | GLM vs Mistral | 41.6% | 0.059 |
| BoN N=64 | Qwen vs Mistral | 44.9% | 0.132 |
| BoN N=64 | **Fleiss κ（5 judges）** | — | **≈0.15** |

## 方法（冻结）

- 同一 prompt 集合、同一生成参数（T=0.7/top_p=0.9/max 256）；judge 只看 Response A/B，不给来源；
- 每个 pair：AB、BA、再随机一票（3 票），多数票；ties 单列；
- 解析失败重试；401/402/403 直接失败，不把错误当 tie；
- judge：DeepSeek-chat、GLM-4.5-air（thinking disabled）、Qwen2.5-7B（本地 vLLM）、Qwen correctness-first、Mistral-7B（本地 vLLM）；
- 扩样本：DPO 566 对（UF holdout 全量）、BoN 300 prompt × N=1/64/128（529 对）；
- 多 seed：DPO800 seed 42/43/44（同配置 800 步）；
- 统计：Wilson 95% CI；ties 不计入分母为主口径；κ 仅作 judge 一致性参考。

## 局限（写报告/面试主动说）

- **人工校准未完成**：150 对盲评表已生成（`p2/human_150.md/csv`），双评按计划暂停；人-judge 一致率/κ 还没有。
- **OpenAI judge 没跑成**：本机与 AutoDL 都连不上 `api.openai.com`（网络超时，无代理）。
- **Mistral 7B 是较弱 judge**：它对 DPO800 给出 61.8%（唯一显著为正），且严格匹配子集 49.5%，更像“长度/格式偏好不稳定”；只作敏感性检查，不当主判据。
- **ties 比例差异大**（Qwen 默认 44% vs Qwen-CF 10%）说明 judge 的“平手阈值”不同；excl-ties 口径对 ties 处理敏感，已同时保留 incl-ties 字段。
- **9 个 seed×judge 格子里仍有 1 个显著负**（DeepSeek seed43），所以不能反过来说“已证明 DPO800 有效”；正确表述是“没有稳健方向”。
- 没有人工大样本、没有 72B 级 judge、没有 BoN 采样 seed。

## 文件

- 汇总：本页 + [judge_summary_v2.csv](judge_summary_v2.csv) + [multi_seed_dpo800.csv](multi_seed_dpo800.csv)；
- 原始：本地 `p2/judge_v2/*.json`（逐对 votes/reason），`p2/judge_v2/raw/`（pairs）；
- 脚本：[judge_api.py](../scripts/judge/judge_api.py)、[generate_pair_outputs_vllm.py](../scripts/judge/generate_pair_outputs_vllm.py)、[aggregate_judges.py](../scripts/judge/aggregate_judges.py)、[make_human_sheet.py](../scripts/judge/make_human_sheet.py)。
