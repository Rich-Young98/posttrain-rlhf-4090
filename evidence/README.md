# evidence/ 说明

> 本目录是"每个数字能溯源"的证据层。先看 [results.md](results.md)；机器可读表是 [run_index.csv](run_index.csv)（34 列 / 26 行）。
> 2026-10-26/27 起，主口径是 **P1-UNIFIED**（统一评测器/同题/3 eval seed）：汇总见 [p1_unified_eval.md](p1_unified_eval.md)，协议定义见 [protocol_matrix.md](protocol_matrix.md)，配置逐项对照见 [config_matrix.md](config_matrix.md)。

## 文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `run_index.csv` | 26 个 run 的机器可读表（34 列：train_seed/model_revision/config_sha256/train_tokens/eval_protocol_id/checkpoint_sha256/result_status…） | 日志 + 评测 JSON + 审计 |
| `results.md` | 一页结果（统一复评 + 项目二 judge） | 汇总 |
| `p1_results.md` | 项目一 GRPO 四组 + RFT + 统一复评 + 审计的总表与读法 | 父目录 P1 日志 |
| `p1_unified_eval.md` | P1-UNIFIED 主表、配对 delta、R1c 重跑（step300/600） | 父目录 `p1/evidence/eval_unified/` |
| `protocol_matrix.md` | 协议定义（P1-LEGACY / P1-NEW / P1-UNIFIED）、可横比规则、58.08 bug 归因 | 父目录 `p1/evidence/` |
| `config_matrix.md` | 4 个历史 run 的 resolved config vs repo 脚本逐字段对照 | 父目录 `p1/evidence/` |
| `artifact_inventory.csv` | 文件级清单（路径/sha256/大小/是否含原始 completion/缺失件） | 父目录 `p1/evidence/` |
| `archive_20261027.md` | 本地 AutoDL 全量归档说明（27.41 GiB：logs/results/checkpoints/adapters/runs/data/scripts） | 2026-10-27 复制 |
| `archive_manifest_20261027.csv` | 归档文件清单（216 行，含 sha256/大小/类别） | 同上 |
| `p1_curves/` | 项目一曲线（R1、R1 vs R1b、R1 家族、MATH-R1、**R1c 重跑 600 步**） | 父目录/本次 run |
| `judge_results.md` | 独立 judge **v1**（DeepSeek 单 judge）方法、CI、长度桶 | 2026-10-25 运行汇总 |
| `judge_results_v2.md` | **多 judge v2**（5 judge × 566 对）：主表、一致性 κ、长度匹配、改写后的结论 | 2026-10-27 |
| `judge_summary_v2.csv` | v2 机器可读表（20 行，逐 judge × 实验，去重后取最大样本） | 同上 |
| `multi_seed_dpo800.csv` | DPO800 多 seed（42/43/44）× 三 judge（Qwen/GLM/DeepSeek）胜率与 CI | 2026-10-27 |
| `judge_summary.csv` | judge 逐组 `a/b/ties/win_rate/CI` | 同上 |
| `rft_results.md` | RFT 的 MATH-500/AIME 对比（3 seed） | `rft-merged_*.json` |
| `dpo_eval_200.json` | DPO 800 vs base 的 RM proxy 评测原始汇总（含长度桶） | `p2/dpo_eval_200.json` |
| `bon_summary.json` | best-of-N N=1..128 的 proxy/gold/长度曲线 | `p2/bon_summary.json` |
| `dpo_manual_review_20.md` | DPO 800 人工复核 20 条 | `p2/dpo_manual_review_20.md` |
| `rm_full_analysis.txt` | RM 全量 2,985 条分析原始输出 | `p2/p2_rm_full_analysis.log` |
| `*.png` | RM 长度偏置、DPO 300 vs 800、BoN 曲线 | 实验图 |

## 口径

- 项目一：**主表只用 P1-UNIFIED**（avg@k + pass@k，3 eval seeds，AIME 双标签 `AIME22–24(90)` / `AIME2024(30 子集)`）；旧 P1-LEGACY/P1-NEW 只作历史，不横比。
- RM：RewardBench accuracy；
- DPO/BoN 的 `RM proxy win`：1.5B UF RM 打分，**不是独立 judge**；
- `judge win`：DeepSeek pairwise，A/B 交换 + 3 票，ties 不计入分母，Wilson 95% CI；
- 长度桶：按 `len(b) - len(a)` 分 `a_longer` / `similar`（±50）/ `b_longer`。

## 缺失件（已声明，不伪造）

- **原 R1c**：checkpoint/adapter 丢失，仍标 `missing_artifact / 未评测`；2026-10-26 另起新轨迹 `p1-r1c-rerun-600` 并评测 step300/600（新 run，不是"补评原 R1c"）。
- 原始逐 pair judge JSON：v1 三份 + v2 多 judge JSON 已拉回本地 `p2/judge_v2/`（并入本地归档 `autodl_archive_20261027` 的可选增量）；本 repo 只放汇总与 CSV。
- 项目一旧协议原始评测 JSON 已从远端拉回父目录 `p1/artifacts/raw_eval/`（含 sha256）；repo 只放汇总。
- 项目一 300 步 DPO 的人工复核未做（800 有完整 20 条）；项目二**人工盲评明确放弃**（2026-10-28 时间取舍）：90/150 对盲评表 + 分析脚本已生成但未标注，证据页/报告写为局限。
