# evidence/ 说明

> 本目录是“每个数字能溯源”的证据层。先看 [results.md](results.md)；机器可读表是 [run_index.csv](run_index.csv)。

## 文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `run_index.csv` | 22 个 run 的指标（项目一 GRPO/RFT + 项目二 RM/DPO/BoN/judge） | 手工汇总自日志/JSON |
| `results.md` | 一页结果（含 judge 与 RFT） | 汇总 |
| `judge_results.md` | 独立 judge 方法、CI、长度桶、局限 | DeepSeek pairwise 运行汇总 |
| `judge_summary.csv` | judge 逐组 `a/b/ties/win_rate/CI` | 同上 |
| `rft_results.md` | RFT 的 MATH-500/AIME 对比（3 seed） | `rft-merged_*.json` |
| `dpo_eval_200.json` | DPO 800 vs base 的 RM proxy 评测原始汇总（含长度桶） | `p2/dpo_eval_200.json` |
| `bon_summary.json` | best-of-N N=1..128 的 proxy/gold/长度曲线 | `p2/bon_summary.json` |
| `dpo_manual_review_20.md` | DPO 800 人工复核 20 条（逐条 base/DPO/RM 分数） | `p2/dpo_manual_review_20.md` |
| `rm_full_analysis.txt` | RM 全量 2,985 条分析原始输出（overall/corr/bins/分组） | `p2/p2_rm_full_analysis.log` |
| `*.png` | RM 长度偏置、DPO 300 vs 800、BoN 曲线 | 实验图 |

## 口径

- AIME/MATH：`avg@k / pass@k`，项目一 base 与 RL 用同一 prompt、同一判分；
- RM：RewardBench accuracy；
- DPO/BoN 的 `RM proxy win`：1.5B UF RM 打分，**不是独立 judge**；
- `judge win`：DeepSeek pairwise，A/B 交换 + 3 票，ties 不计入分母，Wilson 95% CI；
- 长度桶：按 `len(b) - len(a)` 分 `a_longer` / `similar`（±50）/ `b_longer`。

## 缺失件（已声明，不伪造）

- 原始逐 pair judge JSON（`dpo300_judge.json`、`dpo800_judge.json`、`bon_judge.json`）在 AutoDL 实例上，关机后未拉回；本目录保留逐组汇总。
- 项目一原始 `*_eval.json` 仍在远端；repo 里是日志 + 汇总。
- 项目一 300 步 DPO 的人工复核未做（800 有完整 20 条）。
