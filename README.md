# 后训练 RLHF/RLVR 项目仓库（单卡 4090）

> 两个项目：
> 1. **项目一（RLVR）**：单卡 24G 上用 verl + vLLM colocate + LoRA 跑 GRPO，做 DAPO/MATH/lr/步数受控对照，得到可复现负结果 + 工程/数据/评测管线。
> 2. **项目二（RLHF）**：1.5B RM + 自定义 DPO + best-of-N，量化长度偏置、DPO mixed result、held-out RM 同偏置。
>
> 本 repo 是 P0/P1 加固版本：新增独立 judge、RFT（expert iteration）和证据链。原始日志/图在父目录 `../`，完整报告见 `../项目二_最终报告.md` 与 `../后训练RLHF_保姆级执行手册.md` §6.10。

---

## 1. 结果速览

### 项目一：GRPO（Qwen2.5-Math-1.5B + LoRA）

| run | 数据 | 步数 | AIME24 avg/pass | AIME25 avg/pass | MATH-500 avg/pass | train acc |
|---|---|---:|---|---|---|---:|
| base | — | 0 | 7.50 / 10.00 | 9.17 / 20.00 | 72.15 / 82.00 | — |
| R1 | DAPO 5k | 300 | 10.83 / 13.33 | 7.50 / 16.67 | 71.35 / 83.00 | 28.28% |
| R1b | DAPO 5k | 300 (lr 2e-5) | 5.00 / 10.00 | 10.00 / 20.00 | 71.65 / 83.40 | 27.99% |
| MATH-R1 | MATH 7.4k | 300 | 3.33 / 6.67 | 5.00 / 16.67 | 72.65 / 80.40 | 74.20% |

**结论**：没有一致提升；AIME24 +3.3pp 但 AIME25/MATH 反降，R1b 直接掉到 5%。定位为 **受控负结果**：1.5B + LoRA + 300–600 步的学习信号/预算不足。

### 项目二：RM / DPO / best-of-N

| 模块 | 关键数字 | 结论 |
|---|---|---|
| RM | RewardBench 全量 **79.23%**；UF holdout 95.0%；length-only 35.58% | 有真实质量信号 + 强长度偏置 |
| 长度偏置 | chosen 更短 **56.6%**、更长 **99.1%**；Pearson **0.43**；长度匹配 87.6% | "质量 + 长度 shortcut" 并存 |
| DPO 800 | RM proxy 67.5%；长度 **+31.4%**；similar 61.1%；人工 8/20 更差 | mixed / 负结果 |
| DPO 300 | RM proxy 62.0%；长度 +5.5%；similar 55.3% | 更保守，提升弱 |
| best-of-N | N=1→128 proxy -1.94→+1.81、gold -2.28→+1.26；长度 +12.7% | 未复现过优化；held-out RM 同偏置 |

**结论**：不是"对齐成功"，而是把"漂亮指标"拆成长度/风格 shortcut + 真实质量；需要独立 judge。

> 数字全部可在 [evidence/run_index.csv](evidence/run_index.csv) 找到对应 run；原始日志/图在父目录。

---

## 2. 目录结构

```
repo/
  README.md
  requirements.txt
  repro.sh
  scripts/
    p2/                      # 项目二脚本（RM/DPO/BoN/分析，原样拷贝）
    judge/                   # P0 独立 judge：生成配对、pairwise、BoN 配对
    data/prepare_eval_sets.py# MATH-500/AIME24/25 -> JSONL
    eval/eval_math.py        # vLLM 评测：avg@k/pass@k/长度/多 seed
    rft/                     # P1 RFT：采样过滤、LoRA SFT、merge、一键脚本
    verl_pr/                 # 框架二次开发：单卡示例 + sampler + PR 说明
  evidence/
    run_index.csv            # 每个 run 的数字与来源
    results.md               # 一页结果
    *.png                    # RM 长度偏置 / DPO / BoN 曲线
  docs/
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

### P0：独立 judge

```bash
export JUDGE_API_KEY=sk-...
export JUDGE_BASE_URL=https://api.deepseek.com/v1
export JUDGE_MODEL=deepseek-chat

# 生成 base vs DPO 的配对输出
python scripts/judge/generate_pair_outputs.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/runs/dpo-b005/final \
  --prompts_parquet /root/autodl-tmp/work/data/uf_rm/holdout.parquet \
  --n 200 --out /root/autodl-tmp/work/results/dpo800_pairs.jsonl \
  --label_a base --label_b dpo800

# 跑 judge
python scripts/judge/judge_pairwise.py \
  --pairs /root/autodl-tmp/work/results/dpo800_pairs.jsonl \
  --out /root/autodl-tmp/work/results/dpo800_judge.json \
  --backend api --votes 3
```

### P1：RFT vs GRPO

```bash
bash scripts/rft/run_rft.sh
```

细节见 [scripts/rft/README.md](scripts/rft/README.md)、[scripts/judge/README.md](scripts/judge/README.md)。

---

## 5. 证据链

| 项目 | 证据 |
|---|---|
| 项目一 | 父目录 `d4_r1_300.log`、`d4_r1b_lr2e5.log`、`d4_r1c_600.log`、`p1_math_r1_300.log`、`*_curve.png` |
| 项目二 | `scripts/p2/`、父目录 `p2/*.log`、`p2/*.json`、`p2/dpo_manual_review_20.md` |
| 汇总 | [evidence/run_index.csv](evidence/run_index.csv)、[evidence/results.md](evidence/results.md) |

**缺失件**（待补）：项目一原始 `*_eval.json`、项目一远端脚本、wandb 链接。

---

## 6. 局限（面试主动说）

- 只到 1.5B / 单卡；不宣称规模外推；
- 项目一 1 seed；项目二单 seed；
- 项目二原来只有 RM proxy，**本 repo 正在补独立 judge**；
- best-of-N `max_new_tokens=256`，约 35–45% 截断；
- DPO 只跑 β=0.05（300/800 step）。

---

## 7. 状态

- [x] P0 脚本：独立 judge / 配对生成 / BoN 配对
- [x] P0 证据链：`evidence/run_index.csv` + `results.md`
- [x] P1 脚本：RFT 采样/过滤/SFT/merge/eval
- [ ] P0 运行：judge 结果（等 API key）
- [ ] P1 运行：RFT 训练/评测
- [ ] PR：verl 单卡示例 / sampler
