# 项目一 评测协议矩阵（Day 1 冻结版）

> 生成：2026-10-26 · 对应方案 `p1/项目一修补与简历化_保姆级教程_副本.md` §3–§4 Day 1
> 用途：回答"这个数字属于哪个 checkpoint、哪套协议、能不能和另一个数字比"。
> 配套文件：[run_index.csv](run_index.csv) · [artifact_inventory.csv](artifact_inventory.csv) · [config_matrix.md](config_matrix.md) · 原始证据 `p1/artifacts/`

---

## 0. 六个硬结论（Day 1 盘点 + Day 2 统一复评）

| # | 结论 | 证据 |
|---|---|---|
| 1 | 项目一存在**两套不可横比的评测协议**：旧 `d2_tools.py`（MATH-500 base 72.15）与新 `eval_math.py`（MATH-500 base 58.08）。差异至少有 4 项：**prompt 模板、max_tokens、parser、采样 seed/题量**。72.15→58.08 不是模型变化，是协议变化。 | `p1/artifacts/protocol/d2_tools.py`（sha `999ffdf1…`）vs `eval_math_remote.py`（`d6328535…`）；题集比对见 §4 |
| 2 | `r1_300_eval.json` 是**错误 merge 的产物**：与 `base_eval.json` 逐题 responses 完全相同（30/30、30/30、500/500 全等）。R1 的有效评测只有 `r1_300_eval_hf.json`。 | 见 §4.2；两文件 sha 不同，内容逐题相同 |
| 3 | 新协议的 **"AIME24" 实际是 90 题（AIME 2022–2024）**，不是旧协议的 30 题（2024）。所以 `9.49 / 9.77` 与 `7.50 / 10.83` 不可横比；在 2024-only 30 题子集上，新协议 base avg@8 = **10.83**、RFT = **11.67**。 | `data/eval/aime24.jsonl` 90 行；`AI-MO/aimo-validation-aime` |
| 4 | **R1c（600 步）没有任何 checkpoint / adapter / 评测**，只有训练日志。表格必须写"未评测 + 产物缺失"，不能用训练曲线代填。 | `find` 全盘仅 `p1-math-r1-300` 一个 verl checkpoint |
| 5 | **训练 seed 不是干净可控的 42**：resolved config 里 `data.seed=None`（dataloader shuffle 未显式 seed），`trainer.seed` 在 verl v0.8 **不存在**；42 只是 `actor.data_loader_seed` / `fsdp_config.seed` 的默认值。 | `p1/artifacts/configs/*.json`、verl v0.8 `actor.py:178` |
| 6 | **统一复评（Day 2）已完成**：base/R1/R1b/MATH-R1/RFT 用同一 v2 评测器、同题、同解码、同 parser、3 个 eval seed 重跑。MATH-500 所有方法都在 base 的 −0.73 ~ +0.05pp 内；**"R1b AIME 崩 5.00""MATH-R1 AIME 掉 4pp" 在统一协议下消失**（9.81 / 9.17；AIME2024 子集 11.25 / 11.67），旧结论是单 seed 30 题噪声。 | `p1/evidence/eval_unified/` |

> 结论 5 不推翻"负结果"，但把措辞从"seed 42 单 seed"改成"**单次训练、数据顺序未显式 seed**"更准确；面试时这也是一张可打的诚实牌。

---

## 1. 协议定义

### P1-LEGACY —— `d2_tools.py eval`（旧协议，base/R1/R1b/MATH-R1 用）

| 项 | 值 |
|---|---|
| 评测脚本 | `d2_tools.py`，sha256 `999ffdf197b84b213858fdfcdf075dcb4a15feb79bed3d9392efb5d977c7b8aa`（远端 `work/scripts/`，**不在 repo**；已存入 `p1/artifacts/protocol/`） |
| prompt 模板 | AIME：`Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer …`；MATH-500：`… Put your final answer within \boxed{}.`（**带格式指令**） |
| 数据 | AIME24 = `data/raw/aime24` 中 url 含 2024 的 30 题；AIME25 = `data/raw/aime25_alt` 去重 30 题；MATH-500 = `data/raw/math500/test.jsonl` 500 题 |
| n / 采样 | n=4；temperature 0.6；top_p 0.95；max_tokens 3072；max_model_len 4096；max_num_seqs 64；gpu_mem 0.85 |
| eval seed | **未设置**（`SamplingParams` 没传 `seed`，走 vLLM 默认 RNG） |
| 判分器 | verl `default_compute_score`（`aime24`/`aime25` → math_dapo 路由 + 本地 `\boxed` 兜底补丁；MATH-500 → `HuggingFaceH4/MATH-500` 路由） |
| 指标 | `avg@4` = 逐题平均正确率再对题平均；`pass@4` = max(score)>0.5 的题目占比（经验 pass@k） |
| 原始输出 | ✅ 每题存 `responses`（原始 completion）+ `scores` + `ground_truth` |
| 缺陷 | 无 `token_count/truncated/parse_error`；无显式 seed；模板与 parser 强绑定 |

### P1-NEW —— `eval_math.py`（新协议，base-v2 / RFT 用）

| 项 | 值 |
|---|---|
| 评测脚本 | 实跑版本 `eval_math_remote.py` sha256 `d63285357bb4ef0a3a2a6ee33e8cd829ca717db81b39ad23f879c02bf9354e4e`；repo HEAD 版本 `repo/scripts/eval/eval_math.py` sha256 `8c49528bfc88e548bcf807e558dd059085158eb9fae3ebb031f9438e9770b856`（两版逻辑同、函数名不同，远端工作区有未提交改动） |
| prompt 模板 | `apply_chat_template([{role:user, content: problem}])` —— **裸题面，无格式指令** |
| 数据 | `data/eval/math500.jsonl` 500（与旧协议同题同答案）；`aime24.jsonl` **90**（AI-MO/aimo-validation-aime，2022–2024）；`aime25.jsonl` 30（同旧协议题集） |
| n / 采样 | MATH-500 n=4；AIME n=8；T=0.6；top_p 0.95；max_tokens **2048**；max_model_len 4096；enforce_eager=True；gpu_mem 0.85 |
| eval seed | 显式 `42,43,44`，每 seed 单独 `llm.generate`，报 mean/std |
| 判分器 | 正则取**最后一个** `\boxed{…}` → 否则 `final answer|answer` 正则 → 否则**最后一个数字**；再 `math_verify.parse/verify` |
| 指标 | `avg@n`、`pass@1`（pass@k 公式）、`pass@4`/`pass@8`（公式）、`avg_words` |
| 原始输出 | ⚠️ 只存 `preds`（抽取后的答案）、`scores`、`lens`（词数）；**不存原始 completion** → 无法重判 parser |
| 缺陷 | 无 `\boxed` 时退化为"最后一个数字"，误判风险高；不存 raw text；`avg_words` 不是 token |

### P1-TRAIN-GRPO / P1-TRAIN-RFT（训练侧协议，供复现）

| 项 | GRPO（verl v0.8） | RFT（SFT） |
|---|---|---|
| 代码 | verl commit `7aed6b2`（tag v0.8.0）+ **4 个本地补丁**（见 §5） | repo `scripts/rft/*`（commit `d5203b6`） |
| 模型/精度 | Qwen2.5-Math-1.5B-Instruct + LoRA r=32/α=32/all-linear；bf16 | 同 base + LoRA r=32/α=64/dropout0.05；1 epoch |
| 数据 | DAPO 5k（`27316102…`）/ MATH 7.4k（`ec9b9dc1…`） | MATH 7.4k × 8 采样 → `math_verify` 过滤 5,062/7,414（68.3%） |
| 关键超参 | batch 4、rollout n=8、resp 3072、KL loss 0.001、token-mean、lr 1e-5/2e-5、300/600 步 | lr 1e-4 cosine、max_len 2048、317 步、train_runtime 1171s、loss 0.0462 |
| 硬件 | 单卡 4090 24G，18.0 GB reserved，51–71 s/step | 同卡 |

---

### P1-UNIFIED —— `eval_math.py` v2（Day 2 统一复评，当前主口径）

| 项 | 值 |
|---|---|
| 评测脚本 | `repo/scripts/eval/eval_math.py` v2.0，sha256 `41f75d8d2ce33588b0b2d1cac6ecd826f90a248866a4322c7fbd295ecc8aee2a` |
| prompt | `--prompt-mode legacy`（与旧协议同模板，便于对照）；`bare` 只用于 ablation |
| 解码 | T=0.6 / top_p=0.95 / max_new_tokens 3072 / max_model_len 4096 / bf16 / enforce_eager |
| 采样 | MATH-500 n=4；AIME n=8；seeds **42,43,44**（显式传入，逐 seed 单独 generate） |
| 判分 | 提取：平衡 `\boxed{}` → `Answer:/final answer:` 行 → None（**无数字兜底**）；验证：math_verify（gold `\boxed{}`+Latex；pred Expr+Latex） |
| 指标 | avg@n、pass@1、pass@4/8、avg_tokens、parse_fail_rate、truncation_rate |
| 原始输出 | 逐题逐 sample JSONL：completion / parsed_answer / parse_mode / token_count / truncated / parse_error / evaluator_version |
| 结果 | base MATH-500 **75.42±0.29**（旧 P1-LEGACY 72.15）；R1 74.68、R1b 75.12、MATH-R1 75.38、RFT 75.47 |

> P1-UNIFIED 与 P1-LEGACY / P1-NEW 均**不可横比**；只有统一复评内部 5 个模型之间可比。

## 2. 每个数字的归属（run → protocol → artifact）

| run | 训练日志（sha 前 8） | checkpoint / adapter | 评测文件 | 协议 | 有效？ |
|---|---|---|---|---|---|
| base | — | 原始 HF 模型（`model.safetensors` `08faf1e9…`） | `base_eval.json` `5ee0e0fa…` | P1-LEGACY | ✅ |
| R1 (300, lr1e-5) | `d4_r1_300.log` `e6e2439c…` | `p1-r1-300-lora/adapter_model.safetensors` `c6195975…`（merged hf **已丢失**） | `r1_300_eval_hf.json` `d5b42c31…` | P1-LEGACY | ✅ |
| R1（错误 merge） | 同上 | `p1-r1-300`（model_merger 输出 = base 拷贝） | `r1_300_eval.json` `513e58e1…` | P1-LEGACY | ❌ 作废 |
| R1b (300, lr2e-5) | `d4_r1b_lr2e5.log` `ae25ba1d…` | `p1-r1b-lr2e5-lora/…` `cff120da…`（merged hf **已丢失**） | `r1b_300_eval.json` `b5a615d4…` | P1-LEGACY | ✅ |
| R1c (600) | `d4_r1c_600.log` `c96ebf30…` | **无** | **无** | — | ⬜ 缺失 |
| MATH-R1 (300) | `p1_math_r1_300.log` `217f538c…` | `p1-math-r1-lora` `7dc1dd49…` + verl ckpt `6889231b…` + merged `p1-math-r1-hf` `9556ebd4…` | `math_r1_300_eval.json` `00d2c471…` | P1-LEGACY | ✅ |
| MATH-R1 smoke (20) | `p1_math_r1_smoke.log` `d9729e95…` | 未保留 | — | — | 仅环境验证 |
| RFT | `rft.log` `98ad48a4…` | `rft-lora` `7588e101…` + `rft-merged/model.safetensors` `6cbc3221…` | `rft-merged_math500.json` `07c5deaa…`、`_aime24.json` `eacd8634…`、`_aime25.json` `6e23ef89…` | P1-NEW | ✅（仅新协议内可比） |
| base-v2 | — | 同上 base | `Qwen2.5-Math-1.5B-Instruct_math500.json` `0465b91a…`、`_aime24.json` `4bc7b8f8…`、`_aime25.json` `b7bd3ad9…` | P1-NEW | ✅（仅新协议内可比） |
| **base（统一）** | — | base HF | `eval_unified/results/base_math500_legacy.json` 等 3 份 | P1-UNIFIED | ✅ 主口径 |
| **R1（统一）** | 同上 | 本次 PEFT 重 merge `p1-r1-300-hf` | `eval_unified/results/p1-r1-*.json`（3 benchmark） | P1-UNIFIED | ✅ 主口径 |
| **R1b（统一）** | 同上 | 本次 PEFT 重 merge `p1-r1b-lr2e5-hf` | `p1-r1b-*.json` | P1-UNIFIED | ✅ 主口径 |
| **MATH-R1（统一）** | 同上 | `p1-math-r1-hf` | `p1-math-r1-*.json` | P1-UNIFIED | ✅ 主口径 |
| **RFT（统一）** | 同上 | `runs/rft-merged` | `p1-rft-*.json` | P1-UNIFIED | ✅ 主口径 |
| **R1c 重跑 step300** | `p1-r1c-rerun-600.log`（同一次 600 步轨迹） | `p1-r1c-rerun-600/global_step_300` → adapter `a163b4cb…` | `p1/evidence/eval_unified/r1c/p1-r1c-rerun-300-*.json` | P1-UNIFIED | ✅（仅轨迹内） |
| **R1c 重跑 step600** | 同上 | `global_step_600` → adapter `d1f248e8…` | `p1/evidence/eval_unified/r1c/p1-r1c-rerun-600-*.json` | P1-UNIFIED | ✅（仅轨迹内） |

> 完整 34 行（含大小、远端路径、缺失件）见 [run_index.csv](run_index.csv)；文件级清单见 [artifact_inventory.csv](artifact_inventory.csv)。

---

## 3. 可横比矩阵

| 对比 | 结论 | 原因 |
|---|---|---|
| base vs R1/R1b/MATH-R1（P1-LEGACY 内） | ✅ 可比 | 同脚本、同题、同解码参数；但**无显式 eval seed**，只能靠 n=4 内部平均吸收噪声 |
| base-v2 vs RFT（P1-NEW 内） | ✅ 可比 | 同脚本、同题、同解码、同 3 个 seed |
| P1-LEGACY vs P1-NEW | ❌ **禁止横比** | 模板（带/不带格式指令）、max_tokens（3072/2048）、parser、n（4/8）、AIME 题量（30/90）全不同 |
| R1c 与任何 run | ❌ | 无产物、无评测 |
| AIME24（旧 30 题）vs AIME24（新 90 题） | ❌ | 题集不同（新=2022–2024） |
| 同协议内不同 eval seed | ⚠️ 只有 P1-NEW 有（42/43/44）；P1-LEGACY 无 seed，重复评测会变 | — |
| **P1-UNIFIED 内部（base/R1/R1b/MATH-R1/RFT）** | ✅ **唯一推荐横比** | 同脚本、同题、同解码、同 parser、同 3 个 eval seed；配对 bootstrap CI 见 `eval_unified/*_delta.md` |
| P1-UNIFIED vs P1-LEGACY / P1-NEW | ❌ | 判分实现与采样都不同；统一复评的数字不要和 72.15/71.35/58.08 混排 |
| 历史"R1b AIME 5.00" vs 统一"R1b AIME22-24 9.81" | ❌ | 单 seed 30 题 vs 3 seed 90 题，且 parser 不同；只能说明旧值不可靠 |
| AIME22-24(90) vs AIME2024(30 子集) | ⚠️ | 子集是 90 题的真子集、同一次生成派生，可作敏感性分析，但不要当成两个 benchmark 并列汇报 |
| **R1c-rerun step300 vs step600** | ⚠️ 只描述这一条轨迹 | 同 run 的嵌套 checkpoint（lr schedule 按 600 步）；不是独立训练 seed，也不等于独立 300-step run |
| R1c-rerun vs 历史 R1/R1c | ❌ 只能作背景 | 训练随机状态不同；原 R1c 产物已丢失，新 run 是新轨迹 |

---

## 4. 差异归因（为什么 72.15 ≠ 58.08）

### 4.1 题集比对（已逐题验证）

| benchmark | 旧协议 | 新协议 | 交集 | 题目/答案一致？ |
|---|---|---|---|---|
| MATH-500 | 500 | 500 | 500 | ✅ 500/500 题目与 gold 完全一致 |
| AIME25 | 30 | 30 | 30 | ✅ 30/30 |
| AIME24 | 30（2024） | **90（2022–2024）** | 30 | ✅ 旧 30 题全部包含在新 90 题内，答案一致 |

→ 所以 MATH-500 的 72.15 vs 58.08 **不是题集造成的**，是 prompt/parser/解码造成的；AIME24 的差既有协议差、又有题集差（30 vs 90）。

### 4.2 merge bug 的判据

- `r1_300_eval.json` 与 `base_eval.json`：AIME24 30/30、AIME25 30/30、MATH-500 500/500 的 `responses` 列表**完全相同**。
- `r1_300_eval_hf.json`（PEFT 正确 merge）与 base：30/30、30/30、500/500 题的输出**全部不同**（maxdiff 生效）。
- 结论：`verl.model_merger` 对 LoRA checkpoint 只导出 `base + lora_adapter/`，直接评测 = 评 base；必须 `merge_lora_peft.py` 二次合并。这条已写进 `scripts/p1/README.md` §3，但**2026-10-20 12:07 那次评测就是踩了这个坑**，两份 json 都在，正好当证据。

### 4.3 72.15 → 58.08 的归因（Day 2 已实测）

| 因素 | 实测 | 结论 |
|---|---|---|
| 题集 | MATH-500 500/500、AIME25 30/30 逐题相同 | **排除** |
| prompt 模板 | base：legacy 75.42 vs bare 74.88（同解码/parser/seed），Δ **−0.53pp**，CI [−1.67,+0.68] | 只占 ~0.5pp，**不是主因** |
| parser | 旧 `eval_math.py` 的 `\\boxed{([^{}]*)}` 遇嵌套大括号截断（`\\boxed{(3, \\frac{\\pi}{2})}` → `(3, \\`），默认 math_verify 又丢弃元组/符号表达式 → 正确答案判 0；"最后一个数字"兜底再放大误判 | **主因**：v2 parser 下 base = 75.42，比旧 P1-LEGACY 72.15 还高 3.27pp |
| max_tokens | 新协议 2048 vs 3072：AIME 解答更长，2048 下截断更多 | 次要，叠加在 parser 上 |

> "新协议 base 58.08"是**判分链路失效导致的低估**，不能当作"模型变差"或"协议更难"的证据。模板只贡献 ~0.5pp，其余约 13pp 来自 parser（+ 少量 max_tokens）。

---

## 5. verl 本地补丁（框架 delta，简历可写）

| 文件 | sha256（运行期） | 改动 | 为什么 |
|---|---|---|---|
| `verl/utils/reward_score/__init__.py` | `d74d0cd5…` | math_dapo 判分后加 `\boxed{}` 兜底（`pred=="[INVALID]"` 时重判） | Qwen2.5-Math 爱输出 `\boxed{}`，而 math_dapo 只认 `Answer:` → 否则正确解也判 0 |
| `verl/utils/attention_utils.py` | `2d3c03ef…` | flash-attn 缺失时回落 transformers `_unpad_input` / einops | 4090 环境没装 flash-attn |
| `verl/utils/model.py` | `9dadcff3…` | value-head 模型 `flash_attention_2` → `sdpa` | 同上 |
| `verl/workers/config/model.py` | `ba5ede28…` | 默认 `attn_implementation` → `sdpa` | 同上 |

> 补丁 1 直接改变训练 reward（DAPO 的 ±1 判定），必须在报告里披露；补丁 2–4 只影响能否跑起来，不影响数值语义。

---

## 6. Day 2 统一评测冻结清单（✅ 已全部执行，2026-10-26）

- [x] 统一入口 = `repo/scripts/eval/eval_math.py` v2（sha `41f75d8d…`），旧 `d2_tools.py` 只作历史。
- [x] `--prompt-mode {legacy,bare}`，主表用 legacy；bare 只做 MATH-500 全量 ablation（500 题）。
- [x] 解码冻结：T=0.6、top_p=0.95、max_new_tokens=3072、max_model_len=4096。
- [x] seed `42,43,44` 显式传入，所有模型同 seed 列表。
- [x] 数据：MATH-500 500（`0f960a82…`）、AIME22-24 90（`3367410b…`）、AIME25 30（`44e9a5fb…`）；AIME2024(30) 由同一批结果按 ID 派生。
- [x] 逐题 JSONL 已产出（18 个字段，含 completion / parsed / token / truncated / parse_error / evaluator_version）。
- [x] 判分：boxed/answer 提取 + math_verify(verl 兼容配置)，**无数字兜底**；`--parser legacy` 仅供排查。
- [x] 20 题 smoke + 人工抽查 10 条（`eval_unified/parser_spotcheck.md`）后才跑全量。

## 7. 证据文件索引

| 文件 | 位置 | 说明 |
|---|---|---|
| 旧评测器 | `p1/artifacts/protocol/d2_tools.py` | P1-LEGACY 唯一实现，sha `999ffdf1…` |
| 新评测器（实跑版） | `p1/artifacts/protocol/eval_math_remote.py` | sha `d6328535…` |
| 新评测器（repo HEAD） | `repo/scripts/eval/eval_math.py` | sha `8c49528b…`，Day 2 在此改 |
| 评测题集 | `p1/artifacts/protocol/{math500,aime24,aime25}.jsonl` | 500 / 90 / 30 题 |
| 原始评测结果 | `p1/artifacts/raw_eval/*.json`（8 个） | 含逐题 responses/preds/scores |
| 训练 resolved config | `p1/artifacts/configs/*.json`（4 个） | 从日志提取，sha 见 config_matrix |
| 训练指标 | `p1/artifacts/configs/train_metrics.json` | steps/tokens/wall-clock/末段 reward |
| 日志 | 工作区根目录与远端 `work/logs/` | 本地 sha 与远端逐一相同（见 inventory） |
| 统一复评结果 | `p1/evidence/eval_unified/` | 主表、delta、逐题 jsonl（124 MB） |
| R1c 重跑（step300/600） | `p1/evidence/eval_unified/r1c/` | 训练曲线、主表、逐题 jsonl、README |
| 评测器 v2 | `repo/scripts/eval/eval_math.py` | sha `41f75d8d…`；`summarize_eval.py`、`paired_delta.py` 同目录 |
