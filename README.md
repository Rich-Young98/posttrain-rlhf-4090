# 后训练 RLHF/RLVR 项目仓库（单卡 4090）

> **两个项目，一套受控诊断方法：**
> 1. **项目一（RLVR）**：单卡 24G 上用 verl + vLLM colocate + LoRA 跑 GRPO，再做 RFT（expert iteration）；两个方向都得到**可复现负结果**，并有完整工程/数据/评测管线。
> 2. **项目二（RLHF）**：1.5B RM + 自定义 DPO + best-of-N；用**独立 judge** 证伪 DPO（proxy 67.5% → judge 40.9%），同时确认 best-of-N **选择**有效（65.5%/66.7%）。
>
> 原始日志/图在父目录 `../`；完整报告见父目录 `项目一_最终报告.md`、`项目二_最终报告.md`；面试口径见 [docs/项目一_结果与面试口径.md](docs/项目一_结果与面试口径.md)、[docs/项目二_结果与面试口径.md](docs/项目二_结果与面试口径.md)。

**一句话**：项目一用 GRPO + RFT 两条尝试说明“小预算下换 on/off-policy 没有解决问题”（**不能反推主因是容量/数据**，单次训练 + 两套协议无法识别）；项目二用独立 judge 说明“**选择 vs 优化**是两种不同的 reward-hacking 风险”。

---

## 1. 结果速览

### 项目一：GRPO（Qwen2.5-Math-1.5B + LoRA）— 统一复评（P1-UNIFIED）

> 同一评测器 / 同题 / 同解码 / 同 parser / 3 个 eval seed 重跑（2026-10-26）。旧 P1-LEGACY 数字只作历史，不与本表混比。

| run | 数据 | 步数 | MATH-500 avg@4 | AIME22–24(90) avg@8 | AIME2024(30 子集) avg@8 | AIME25(30) avg@8 | train acc |
|---|---|---:|---:|---:|---:|---:|---:|
| base | — | 0 | **75.42±0.29** | **9.35±0.29** | 10.56±0.86 | **10.56±0.52** | — |
| R1 | DAPO 5k | 300 | 74.68±0.39 | 9.12±0.24 | 10.83±0.68 | 9.44±0.39 | 28.28% |
| R1b | DAPO 5k | 300 (lr 2e-5) | 75.12±0.13 | 9.81±0.57 | 11.25±1.48 | 9.44±0.39 | 27.99% |
| R1c（原） | DAPO 5k | 600 | 未评测 | 未评测 | 未评测 | 未评测 | 29.43% |
| R1c-rerun step300 | DAPO 5k | 600 轨迹的中间点 | 75.12±0.35 | 9.54±0.51 | 11.11±0.86 | 9.31±1.75 | — |
| R1c-rerun step600 | 同轨迹 | 600 | 75.33±0.26 | 8.75±0.11 | 10.69±0.20 | 10.69±1.42 | — |
| MATH-R1 | MATH 7.4k | 300 | 75.38±0.57 | 9.17±0.41 | 11.67±0.68 | 10.42±1.23 | 74.20% |
| RFT | MATH 7.4k → 5,062 | 317 SFT | **75.47±0.08** | 9.35±0.35 | 10.42±0.59 | **6.94±0.39** | — |

**结论**：没有任何方法带来稳定提升（MATH-500 全部在 base 的 −0.73 ~ +0.05pp；R1 的 −0.73pp CI 不含 0，即"略差"）。**R1c 重跑**（新 600 步轨迹，step300 vs step600）MATH-500 差 +0.22pp [−0.43,+0.90]，AIME 方向不一致 → 单次轨迹未见稳定 300→600 变化。**旧结论"R1b AIME 崩到 5.00""MATH-R1 AIME 掉 4pp"在统一协议下消失**（9.81 / 9.17，AIME2024 子集 11.25 / 11.67）→ 旧值是单 seed 30 题噪声。RFT 的负效应只在 AIME25（−3.61pp，CI [−7.08, −0.56]）。主因（容量/数据/reward/有效 rollout/预算）仍是待验证假设，不是已证明结论。

> 逐 run 读法、delta、逐题 jsonl：[p1/evidence 的 eval_unified](../p1/evidence/eval_unified/README.md)；证据页：[evidence/p1_results.md](evidence/p1_results.md)；曲线：[evidence/p1_curves/](evidence/p1_curves/)；启动脚本：[scripts/p1/](scripts/p1/)；评测器：[scripts/eval/](scripts/eval/)。

### 项目二：RM / DPO / best-of-N

| 模块 | 关键数字 | 结论 |
|---|---|---|
| RM | RewardBench 全量 **79.23%**；UF holdout 95.0%；length-only 35.58%；长度匹配 87.6% | 真实质量信号 + 强长度偏置 |
| 长度偏置 | chosen 更短 **56.6%**、更长 **99.1%**；Pearson **0.43** | "质量 + 长度 shortcut" 并存 |
| DPO 800 | RM proxy **67.5%**；独立 judge **40.9%**（CI 33.8–48.4）；长度 **+31.4%** | **负结果**：proxy 高估 26.6pp |
| DPO 300 | RM proxy 62.0%；独立 judge **52.9%**（CI 44.6–61.0，平手）；长度 +5.5% | 与 base 无显著差异 |
| best-of-N 过优化 | N=1→128 proxy/gold 均上升；N=64→128 gold +0.31；长度 +12.7% | 未复现过优化（negative） |
| best-of-N 选择 | 独立 judge N=64/128 = **65.5%/66.7%**；similar 约 66% | **选择有效（正结果）** |

> 每个数字可在 [evidence/run_index.csv](evidence/run_index.csv) 找到对应 run；judge 方法/CI 见 [evidence/judge_results.md](evidence/judge_results.md)。

---

## 2. 目录结构

```
repo/
  README.md
  requirements.txt
  repro.sh
  scripts/
    p1/                      # 项目一：GRPO 启动脚本 + LoRA 合并（按实跑日志重建）
    p2/                      # 项目二：RM / 自定义 DPO / best-of-N / 分析
    judge/                   # P0 独立 judge：配对生成、pairwise、BoN 配对、分析
    data/prepare_eval_sets.py# MATH-500/AIME24/25 -> JSONL
    eval/eval_math.py        # vLLM 评测：avg@k/pass@k/长度/多 seed
    rft/                     # P1 RFT：采样过滤、LoRA SFT、merge、一键脚本
    verl_pr/                 # 框架二次开发：单卡示例 + sampler + PR 说明
  evidence/
    run_index.csv            # 每个 run 的数字与来源（22 行）
    results.md               # 一页结果
    p1_results.md            # 项目一：GRPO + RFT 总表/逐 run/证据
    p1_curves/               # 项目一：R1 / R1b / R1 家族 / MATH-R1 曲线
    judge_results.md         # 独立 judge 方法/CI/长度桶
    judge_summary.csv        # judge 逐组 a/b/ties（机器可读）
    rft_results.md           # RFT 负结果
    dpo_eval_200.json        # DPO 生成评测原始汇总
    bon_summary.json         # best-of-N 曲线原始汇总
    dpo_manual_review_20.md  # DPO 人工复核 20 条
    rm_full_analysis.txt     # RM 全量分析原始输出
    *.png                    # RM 长度偏置 / DPO / BoN 曲线
  docs/
    项目一_结果与面试口径.md   # 项目一：一页结果 + 30s/2min 讲法
    项目二_结果与面试口径.md   # 项目二：一页结果 + 30s/2min 讲法 + 追问
    D6_启动命令.md
    方案B_自定义DPO.md
```

---

## 3. 环境

已验证（AutoDL 4090 24G，venv）：

```
torch 2.8.0 + cu128
vllm 0.11.0
transformers 4.57.1 / tokenizers 0.22.1 / huggingface_hub 0.34.4
datasets 3.6.0 / peft 0.17.1 / accelerate 1.10.1
numpy 1.26.4 / pyarrow 21.0.0
verl v0.8.0（训练）
```

```bash
source /root/autodl-tmp/venv-verl/bin/activate
pip install -r requirements.txt
```

---

## 4. 快速开始

### 4.1 项目一：单卡 GRPO

```bash
# R1：DAPO 5k，300 步，lr 1e-5
MODE=dapo STEPS=300 LR=1e-5 EXP=p1-r1-300 bash scripts/p1/run_grpo.sh

# MATH-R1：MATH 7.4k
MODE=math STEPS=300 LR=1e-5 EXP=p1-math-r1-300 bash scripts/p1/run_grpo.sh
```

合并 LoRA（**不能直接评 model_merger 输出**）：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir .../p1-r1-300/global_step_300/actor \
  --target_dir /root/autodl-tmp/work/merged/p1-r1-300

python scripts/p1/merge_lora_peft.py \
  --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/merged/p1-r1-300/lora_adapter \
  --out /root/autodl-tmp/work/merged/p1-r1-300-hf
```

细节、显存坑、评测命令见 [scripts/p1/README.md](scripts/p1/README.md)；结果见 [evidence/p1_results.md](evidence/p1_results.md)。

### 4.2 项目二：独立 judge（最关键的一步）

```bash
export JUDGE_API_KEY=sk-...                     # 不要提交到 repo
export JUDGE_BASE_URL=https://api.deepseek.com/v1
export JUDGE_MODEL=deepseek-chat

# 1) 生成 base vs DPO 的配对输出（需要 GPU）
python scripts/judge/generate_pair_outputs.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/runs/dpo-b005/final \
  --prompts_parquet /root/autodl-tmp/work/data/uf_rm/holdout.parquet \
  --n 200 --out /root/autodl-tmp/work/results/dpo800_pairs.jsonl \
  --label_a base --label_b dpo800

# 2) 跑 judge（A/B 交换 + 3 票）
python scripts/judge/judge_pairwise.py \
  --pairs /root/autodl-tmp/work/results/dpo800_pairs.jsonl \
  --out /root/autodl-tmp/work/results/dpo800_judge.json \
  --backend api --votes 3

# 3) 跑 best-of-N 配对 + judge
python scripts/judge/judge_bon_pairs.py \
  --selected /root/autodl-tmp/work/results/bon128_selected.jsonl \
  --ns 64 128 --out /root/autodl-tmp/work/results/bon_pairs.jsonl
python scripts/judge/judge_pairwise.py \
  --pairs /root/autodl-tmp/work/results/bon_pairs.jsonl \
  --out /root/autodl-tmp/work/results/bon_judge.json \
  --backend api --votes 3

# 4) 分桶分析
python scripts/judge/analyze_judge.py --judge_json /root/autodl-tmp/work/results/dpo800_judge.json
```

一键版（DPO 300/800 + BoN）：

```bash
bash scripts/judge/run_judge.sh
bash scripts/judge/run_judge_bon.sh
```

细节与踩坑见 [scripts/judge/README.md](scripts/judge/README.md)。

### 4.3 RFT / expert iteration

```bash
bash scripts/rft/run_rft.sh
```

流程：MATH 7.4k × 8 采样 → `math_verify` 过滤（kept 5,062 / 68.3%）→ 最短正确回答 LoRA SFT（317 步）→ merge → 3 seed 评测。详见 [scripts/rft/README.md](scripts/rft/README.md)。

> verl 单卡示例/PR 预案仍保留在 [scripts/verl_pr/](scripts/verl_pr/)，用于后续提 PR；项目一实跑配置以 [scripts/p1/](scripts/p1/) 为准。

---

## 5. 证据链

| 项目 | 证据 |
|---|---|
| 项目一 GRPO | [evidence/p1_results.md](evidence/p1_results.md)、[evidence/p1_curves/](evidence/p1_curves/)、父目录 `d4_r1_300.log` 等；启动脚本 [scripts/p1/](scripts/p1/) |
| 项目一 RFT | [evidence/rft_results.md](evidence/rft_results.md)、[scripts/rft/](scripts/rft/)、`run_index.csv` 后 6 行 |
| 项目二 RM | [evidence/rm_full_analysis.txt](evidence/rm_full_analysis.txt)、`rm_length_bias.png`、父目录 `p2/*.log` |
| 项目二 DPO | [evidence/dpo_eval_200.json](evidence/dpo_eval_200.json)、[evidence/dpo_manual_review_20.md](evidence/dpo_manual_review_20.md)、[evidence/dpo_300_vs_800.png](evidence/dpo_300_vs_800.png) |
| 项目二 best-of-N | [evidence/bon_summary.json](evidence/bon_summary.json)、[evidence/bon_curve.png](evidence/bon_curve.png)、[evidence/bon128_curve.png](evidence/bon128_curve.png) |
| 独立 judge | [evidence/judge_results.md](evidence/judge_results.md)、[evidence/judge_summary.csv](evidence/judge_summary.csv)、`scripts/judge/` |
| 汇总 | [evidence/run_index.csv](evidence/run_index.csv)（22 行）、[evidence/results.md](evidence/results.md) |

**证据边界（主动说）**：原始逐 pair judge JSON（约 0.7MB/个）产生在 AutoDL `/root/autodl-tmp/work/results/`，实例关机后未拉回本地；repo 内保留逐组 `a/b/ties`、CI、长度桶汇总，数字与报告一致。项目一原始 `*_eval.json` 和训练 launch 脚本也仍在远端；repo 的 `scripts/p1/` 是按日志 resolved config 重建的启动脚本。

---

## 6. 局限（面试主动说）

- 只到 1.5B / 单卡；不宣称规模外推；
- 项目一 1 seed 训练，AIME 评测 n=4/8；R1c 未评测；
- 项目二单 seed；独立 judge 只有 DeepSeek 一个、DPO 200 对 / BoN 89+89 对；未做人工大样本或 7B judge 交叉验证；
- best-of-N `max_new_tokens=256`，约 35–45% 截断；
- DPO 只跑 β=0.05（300/800 step 消融），β 未全扫；
- 项目一 MATH-500 有两套评测协议（72.15 vs 58.08），不横比；
- 没有 AlpacaEval LC，不声称 LC win rate。

---

## 7. 状态

- [x] 项目一：GRPO 四组 + RFT 全流程、结果表、曲线、启动/合并脚本（负结果）
- [x] P0 脚本：独立 judge / 配对生成 / BoN 配对 / 分桶分析
- [x] P0 运行：judge 结果（DPO 300/800、BoN 64/128）
- [x] P0 证据链：`run_index.csv` + `results.md` + `p1_results.md` + `judge_results.md` + 原始汇总 JSON
- [x] P1 脚本 + 运行：RFT 采样/过滤/SFT/merge/eval（负结果）
- [x] 报告/面试：`docs/项目一_结果与面试口径.md`、`docs/项目二_结果与面试口径.md`
- [ ] PR：verl 单卡示例 / sampler（`scripts/verl_pr/` 已备好，待提）
