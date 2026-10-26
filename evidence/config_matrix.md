# 项目一 resolved config 对照（日志 vs repo 脚本）

> Day 1 交付物 · 2026-10-26
> 目的：回答"每个数字属于哪个 checkpoint、哪个配置"。verl **没有落盘 resolved YAML**，但每个训练日志开头都打印了完整 resolved config，本文件即从日志解析出的真相，与 repo 里"按日志重建"的 `scripts/p1/run_grpo.sh` 逐项对照。
> 数据：[configs/](configs/) 4 个 JSON + [train_metrics.json](configs/train_metrics.json) · 提取脚本：[extract_config.py](extract_config.py)、[extract_metrics.py](extract_metrics.py)

---

## 0. 三个必须知道的发现

1. **`scripts/p1/run_grpo.sh` 有 3 个字段与实跑不一致**：`use_shm`（脚本 True / 实跑 False）、`rollout.load_format`（safetensors / dummy）、`rollout.layered_summon`（True / False）。照抄脚本重跑 ≠ 复现原 run；应以日志为准。
2. **`trainer.seed=42` 在 verl v0.8 不存在**：resolved config 的 `trainer` 没有 `seed` 字段（`grep` 过 `verl/trainer/config/*.py` 无此字段）。照抄脚本会报 Hydra 覆盖错误。真实与 seed 相关的字段是 `actor.data_loader_seed=42`（默认值）与 `fsdp_config.seed=42`，而 **`data.seed=None`** → dataloader 的 shuffle 顺序没有显式 seed。
   → 简历/报告里"seed 42 单 seed"应改为"**单次训练；数据顺序未显式 seed**"。
3. **`save_freq` 也与脚本不同**：R1 实跑 `save_freq=100`（脚本写 `$STEPS`=300）。原 run 应有 step 100/200 的 ckpt，磁盘上只剩 300（空间清理），不影响 300 步结果，但"只保存最终 ckpt"这句话是错的。

---

## 1. 字段对照表

图例：✅ = 日志可证实且与脚本一致；❌ = 日志可证实但与脚本**不一致**；❓ = 日志无此字段 / 无法证实。

| 参数 | run_grpo.sh | R1 (`d4_r1_300`) | R1b | R1c | MATH-R1 | 证据 |
|---|---|---|---|---|---|---|
| data.train_files | `$WORK/data/{dapo_5k,math_train}_verl/train.parquet` | dapo 5k | dapo 5k | dapo 5k | math 7.4k | ✅ |
| data.val_files | 同 train | 同 train | 同 train | 同 train | 同 train | ✅（等于拿训练集当 val，`test_freq=-1` 实际不跑） |
| data.train_batch_size | 4 | 4 | 4 | 4 | 4 | ✅ |
| data.max_prompt_length | 1024 | 1024 | 1024 | 1024 | 1024 | ✅ |
| data.max_response_length | 3072 | 3072 | 3072 | 3072 | 3072 | ✅ |
| data.shuffle | （默认） | True | True | True | True | ✅ |
| data.seed | 未设 | **None** | None | None | None | ❌（脚本未涉及，但结论是"未显式 seed"） |
| truncation | （默认 error） | error | error | error | error | ✅ |
| filter_overlong_prompts | （默认 false） | false | false | false | false | ✅ |
| model.path | Qwen2.5-Math-1.5B-Instruct | 同 | 同 | 同 | 同 | ✅ |
| model.lora_rank | 32 | 32 | 32 | 32 | 32 | ✅ |
| model.lora_alpha | 32 | 32 | 32 | 32 | 32 | ✅ |
| model.target_modules | all-linear | all-linear | all-linear | all-linear | all-linear | ✅ |
| model.use_shm | **True** | **False** | False | False | False | ❌ 脚本错 |
| rollout.name | vllm | vllm | vllm | vllm | vllm | ✅ |
| rollout.n | 8 | 8 | 8 | 8 | 8 | ✅ |
| rollout.temperature / top_p / top_k | （默认 1.0 / 1 / -1） | 1.0 / 1 / -1 | 同 | 同 | 同 | ✅ 训练采样温度 = 1.0 |
| rollout.load_format | **safetensors** | **dummy** | dummy | dummy | dummy | ❌ 脚本错 |
| rollout.layered_summon | **True** | **False** | False | False | False | ❌ 脚本错 |
| rollout.gpu_memory_utilization | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | ✅ |
| rollout.response_length / max_model_len | （3072）/ 4096 | 3072 / 4096 | 同 | 同 | 同 | ✅ |
| rollout.tensor_model_parallel_size | 1 | 1 | 1 | 1 | 1 | ✅ |
| rollout.log_prob_micro_batch_size_per_gpu | 1 | 1 | 1 | 1 | 1 | ✅ |
| ref.log_prob_micro_batch_size_per_gpu | 1 | 1 | 1 | 1 | 1 | ✅ |
| actor.optim.lr | `$LR` | 1e-5 | 2e-5 | 1e-5 | 1e-5 | ✅ |
| actor.ppo_mini_batch_size | 4 | 4 | 4 | 4 | 4 | ✅ |
| actor.ppo_micro_batch_size_per_gpu | 1 | 1 | 1 | 1 | 1 | ✅ |
| actor.ppo_epochs | （默认 1） | 1 | 1 | 1 | 1 | ✅ |
| actor.use_kl_loss | True | True | True | True | True | ✅ |
| actor.kl_loss_coef | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | ✅ |
| actor.kl_loss_type | （默认 low_var_kl） | low_var_kl | low_var_kl | low_var_kl | low_var_kl | ✅ |
| actor.loss_agg_mode | token-mean | token-mean | token-mean | token-mean | token-mean | ✅ |
| actor.clip_ratio | （默认 0.2） | 0.2 | 0.2 | 0.2 | 0.2 | ✅ |
| actor.entropy_coeff | （默认 0） | 0 | 0 | 0 | 0 | ✅ |
| actor.grad_clip | （默认 1.0） | 1.0 | 1.0 | 1.0 | 1.0 | ✅ |
| actor.fsdp dtype | （默认 bf16） | bf16 | bf16 | bf16 | bf16 | ✅ |
| algorithm.adv_estimator | grpo | grpo | grpo | grpo | grpo | ✅ |
| algorithm.use_kl_in_reward | 未设 | False | False | False | False | ✅（KL 走 loss 不走 reward） |
| algorithm.norm_adv_by_std_in_grpo | 未设 | True | True | True | True | ✅ |
| trainer.logger | console | console | console | console | console | ✅ |
| trainer.n_gpus_per_node / nnodes | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | ✅ |
| trainer.total_epochs | 1 | 1 | 1 | 1 | 1 | ✅ |
| trainer.total_training_steps | `$STEPS` | 300 | 300 | 600 | 300 | ✅ |
| trainer.save_freq | `$STEPS` | **100** | 300 | 600 | 300 | ❌ R1 不一致 |
| trainer.test_freq | -1 | -1 | -1 | -1 | -1 | ✅ |
| trainer.val_before_train | False | False | False | False | False | ✅ |
| trainer.project_name / experiment_name | rlvr-interview / `$EXP` | p1-r1-300 | p1-r1b-lr2e5 | p1-r1c-600 | p1-math-r1-300 | ✅ |
| trainer.seed | **42** | **字段不存在** | 不存在 | 不存在 | 不存在 | ❌ 脚本错（v0.8 无此字段） |
| critic.model.path | — | `~/models/deepseek-llm-7b-chat`（GRPO 不建 critic，无影响） | 同 | 同 | 同 | ✅（配置残留） |

---

## 2. 证据等级（按方案 §4 Day1 第 4 条口径）

| 类别 | 字段 | 说明 |
|---|---|---|
| **日志可证实** | 上表标 ✅ 的全部 | resolved config 直接打印在日志里，提取自 `p1/artifacts/configs/*.json` |
| **由脚本重建** | 启动命令本身、环境变量（`PY/VERL/WORK`）、log 路径 | 原 launch 命令只在 AutoDL shell/tmux 里，未入库；脚本是"按日志重建" |
| **未知** | 原始启动命令、是否显式传过 `data.seed`、`trainer.seed` 是否存在过、step 100/200 ckpt 是否曾保存 | 日志无法回答；重跑时按 Day 2 冻结版固定 |

---

## 3. 每 run 实跑规模（从日志解析，非估算）

| run | steps | 总 tokens（prompt+response） | response tokens | wall-clock | s/step | 末 20 步 reward | 折合 acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 300 | 9,445,717 | 7,769,117 | 5.94 h | 71.25 | -0.566 | 28.28%（±1 尺度） |
| R1b | 300 | 9,507,806 | 7,831,206 | 5.55 h | 66.55 | -0.500 | 27.99% |
| R1c | 600 | 18,719,144 | 15,376,936 | 9.37 h | 56.22 | -0.466 | 29.43% |
| MATH-R1 | 300 | 6,746,524 | 5,474,644 | 4.27 h | 51.23 | 0.725 | 74.20%（0/1 尺度） |
| MATH-R1 smoke | 20 | 443,849 | 360,953 | 0.29 h | 51.55 | 0.731 | 73.12% |
| RFT（SFT） | 317 | n/a（SFT 未记 token） | — | 0.33 h（1171 s） | 3.69 | loss 0.0462 | — |

> 峰值显存：`actor/perf/max_memory_reserved_gb = 18.01`，`max_memory_allocated_gb = 17.96`（R1），与报告"18GB reserved"一致。
> `pg_clipfrac` 全程 ≈ 0、`response_length/clip_ratio` ≤ 0.15%，说明 300–600 步内策略更新幅度很小 —— 这是"不是 lr 太猛"的日志证据，但**不能**反推"瓶颈是容量"。

---

## 4. 复现校验命令

```bash
# 1) 重新提取某 run 的 resolved config（本地已有日志）
cd ~/面试项目准备/p1
python3 artifacts/extract_config.py ~/面试项目准备/d4_r1_300.log artifacts/configs/d4_r1_300.json

# 2) 校验日志哈希与远端一致（同一条命令在两端跑，结果应相同）
shasum -a 256 d4_r1_300.log
ssh <autodl> 'sha256sum /root/autodl-tmp/work/logs/d4_r1_300.log'

# 3) 校验 config 与日志一致（改脚本后重跑前建议做）
diff <(python3 -c "import json;print(json.dumps(json.load(open('artifacts/configs/d4_r1_300.json')),sort_keys=True,indent=1))") \
     <(python3 -c "...")
```

---

## 5. Day 2 要修的脚本问题（登记，不在此次改）

- [ ] 删掉 `trainer.seed=42`（v0.8 无此字段）；改用 `data.seed=42`（真正给 dataloader shuffle 用）+ `actor_rollout_ref.actor.data_loader_seed=42`。
- [ ] `use_shm=False`、`rollout.load_format=dummy`、`layered_summon=False` 与实跑对齐（或写明"重建版有意改成更适合单卡的设置"）。
- [ ] `save_freq` 参数化，别再和 `STEPS` 绑死。
- [ ] 启动时把 `git -C $VERL rev-parse HEAD`、4 个补丁的 sha、`pip freeze` 一起写进 log 第一行。
