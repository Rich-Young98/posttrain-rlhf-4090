# 项目一：单卡 GRPO（verl v0.8 + vLLM 0.11 + LoRA）

> 结果总表：[../../evidence/p1_results.md](../../evidence/p1_results.md)；一页面试口径：[../../docs/项目一_结果与面试口径.md](../../docs/项目一_结果与面试口径.md)。
> 本目录的启动脚本是**按实跑日志里的 resolved config 重建**的（原始 launch 命令在 AutoDL 上，未入库）。
>
> ⚠️ **2026-10-26 逐项核对结果**：脚本与实跑日志有 **4 处不一致**，照抄不能等同于复现原 run：
> ① `use_shm`：脚本 True / 实跑 **False**；② `rollout.load_format`：脚本 `safetensors` / 实跑 **`dummy`**；
> ③ `rollout.layered_summon`：脚本 True / 实跑 **False**；④ `trainer.seed=42`：**verl v0.8 没有这个字段**（照抄会报错），
> 实跑里 `data.seed=None`（数据顺序未显式 seed），42 只是 `actor.data_loader_seed` / `fsdp_config.seed` 的默认值。
> 逐字段对照见父目录 `p1/evidence/config_matrix.md`。下次重跑前请先按 Day 2 冻结版改这 4 处。

## 1. 环境

```bash
source /root/autodl-tmp/venv-verl/bin/activate
# torch 2.8.0+cu128 / vllm 0.11.0 / verl v0.8.0 / peft 0.17.1
```

数据与模型：

```
/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct
/root/autodl-tmp/work/data/dapo_5k_verl/train.parquet    # DAPO 5k, data_source=math_dapo
/root/autodl-tmp/work/data/math_train_verl/train.parquet # MATH 7.4k, data_source=...MATH-lighteval
```

数据准备原始脚本（远端，不在本 repo）：`prepare_v2.py`（DAPO 去重/去污染/抽 5k）、`prepare_math.py`（MATH 7.4k）；逻辑见父目录手册 §4.4–4.5。

## 2. 跑 GRPO

```bash
# R1：DAPO 5k，300 步，lr 1e-5
MODE=dapo STEPS=300 LR=1e-5 EXP=p1-r1-300 bash scripts/p1/run_grpo.sh

# R1b：只改 lr
MODE=dapo STEPS=300 LR=2e-5 EXP=p1-r1b-lr2e5 bash scripts/p1/run_grpo.sh

# R1c：只改步数
MODE=dapo STEPS=600 LR=1e-5 EXP=p1-r1c-600 bash scripts/p1/run_grpo.sh

# MATH-R1：换数据
MODE=math STEPS=300 LR=1e-5 EXP=p1-math-r1-300 bash scripts/p1/run_grpo.sh
```

关键超参（与日志一致）：`train_batch_size=4`、`rollout.n=8`、`max_prompt_length=1024`、
`max_response_length=3072`、LoRA rank/alpha=32、`all-linear`、KL loss 0.001、`token-mean`、
`gpu_memory_utilization=0.45`、单卡 FSDP、seed 42。

## 3. 合并 LoRA（必须做对）

`verl.model_merger` 对 LoRA checkpoint **只导出 base + `lora_adapter/`**，直接评测会得到与 base 完全相同的分数。先取 adapter，再用 PEFT 合并：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /root/autodl-tmp/work/code/verl/checkpoints/rlvr-interview/p1-r1-300/global_step_300/actor \
  --target_dir /root/autodl-tmp/work/merged/p1-r1-300

mv /root/autodl-tmp/work/merged/p1-r1-300/lora_adapter \
   /root/autodl-tmp/work/merged/p1-r1-300-lora
rm -rf /root/autodl-tmp/work/merged/p1-r1-300

python scripts/p1/merge_lora_peft.py \
  --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/merged/p1-r1-300-lora \
  --out /root/autodl-tmp/work/merged/p1-r1-300-hf
```

脚本会打印 `q_proj/o_proj/down_proj` 的 `maxdiff`；**必须 > 0**，否则你合并的是 base。

## 4. 评测

项目一原始评测用远端 `d2_tools.py`（`base_eval.json` / `r1_300_eval_hf.json` / `r1b_300_eval.json` / `math_r1_300_eval.json`，未入库）。repo 里的通用评测器需要先把 benchmark 转 JSONL：

```bash
python scripts/data/prepare_eval_sets.py \
  --datasets math500,aime24,aime25 \
  --out /root/autodl-tmp/work/data/eval

python scripts/eval/eval_math.py \
  --model /root/autodl-tmp/work/merged/p1-r1-300-hf \
  --data /root/autodl-tmp/work/data/eval/math500.jsonl \
  --n 4 --seeds 42,43,44 --max-new-tokens 3072 \
  --out /root/autodl-tmp/work/results/p1-r1-300_math500.json
```

> `scripts/eval/eval_math.py` 是后来为 RFT 写的 3-seed 评测器；与项目一老评测协议（`d2_tools.py`）的 MATH-500 base 数不同（72.15 vs 58.08），**不要横比**。

## 5. 结果（摘要）

| run | 数据 | 步数 | lr | train acc | AIME24 avg/pass | AIME25 avg/pass | MATH-500 avg/pass |
|---|---|---:|---:|---:|---|---|---|
| base | — | 0 | — | — | 7.50 / 10.00 | 9.17 / 20.00 | 72.15 / 82.00 |
| R1 | DAPO 5k | 300 | 1e-5 | 28.28% | 10.83 / 13.33 | 7.50 / 16.67 | 71.35 / 83.00 |
| R1b | DAPO 5k | 300 | 2e-5 | 27.99% | 5.00 / 10.00 | 10.00 / 20.00 | 71.65 / 83.40 |
| R1c | DAPO 5k | 600 | 1e-5 | 29.43% | 未评测 | 未评测 | 未评测 |
| MATH-R1 | MATH 7.4k | 300 | 1e-5 | 74.20% | 3.33 / 6.67 | 5.00 / 16.67 | 72.65 / 80.40 |

**结论**：加 lr 只增大 KL；加步数仍卡 ~30%；换 MATH 让 train acc 到 74% 但 eval 不涨；**没有观察到跨评测集一致提升**（不能说"已证明瓶颈在容量/数据/预算"）。

> 本表全部是**旧协议**（AIME24 = 2024-only 30 题，MATH-500 带 `\boxed{}` 指令、3072 tokens、无显式 eval seed）。RFT / base-v2 用的是新协议（**AIME24 = 90 题，2022–2024**、裸题面、2048 tokens、seeds 42/43/44），两套数字不可横比。

## 6. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `No available memory for cache blocks` | vLLM 分到的显存太少 | **提高** `gpu_memory_utilization`（0.35→0.45） |
| 训练侧 CUDA OOM | 训练占太多 | 降 batch / response / rollout.n，或降 vLLM 比例 |
| 评测分数和 base 一模一样 | 直接评了 model_merger 输出（base 拷贝） | 用 PEFT 合并 adapter，先看 maxdiff |
| `No module named 'ray'` | tmux 新 shell 没 activate | 用绝对路径 python |
| exit 时 `DataLoader worker Killed` | Ray 清理噪音 | 不影响结果 |
