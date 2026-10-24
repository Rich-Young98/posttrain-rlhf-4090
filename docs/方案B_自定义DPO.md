# 方案 B：自定义 DPO（不依赖 TRL）

> 项目二 D8 的实际执行方案。目标：在 24G 4090 上，用 Qwen2.5-1.5B-Instruct + LoRA，对 UltraFeedback 做 DPO，β=0.05，并把长度变化、reward margin、KL 全部记录清楚。
>
> 为什么不用 TRL：当前环境是 transformers 4.57.1 + peft 0.17.1，TRL 版本兼容性不确定；装错会把 verl 环境升坏。自定义 DPO 只有 policy + frozen reference + LoRA，24G 够，日志可控，面试也能讲清每个张量。

---

## 1. DPO 在做什么

给定偏好 pair \((x, y_w, y_l)\)，DPO 不训练 RM，而是直接提高 chosen 相对 rejected 的概率，同时用 reference 模型约束 policy 不要跑太远。

\[
L = -\log \sigma\left(\beta \left[
(\log \pi_\theta(y_w|x) - \log \pi_{ref}(y_w|x))
- (\log \pi_\theta(y_l|x) - \log \pi_{ref}(y_l|x))
\right]\right)
\]

- \(\pi_\theta\)：policy，Qwen2.5-1.5B-Instruct + LoRA；
- \(\pi_{ref}\)：frozen base，不加 LoRA；
- \(\beta=0.05\)：越小越保守；
- 标准 DPO 用 **response token logprob 之和**，不要按 token 数平均；length-normalized 是另一套目标，要单独命名。

---

## 2. 数据

- 输入：`/root/autodl-tmp/work/data/uf_rm/train.parquet`；
- 字段：`prompt`、`chosen`、`rejected`；
- 训练量：先 10k，跑通后可加到 27k；
- 留出：`holdout.parquet` 566 条，prompt-disjoint；
- 长度基线：chosen 更长 64.39%，平均 +91 token；DPO 后长度很可能上升，必须记录。

---

## 3. 模型与显存

| 组件 | 说明 | 显存 |
|---|---|---|
| policy | Qwen2.5-1.5B-Instruct + LoRA | ~3.1GB 权重 + LoRA |
| reference | 同一 base，bf16，冻结 | ~3.1GB |
| activations | seq 2048，bs 1，grad ckpt | ~8–12GB |
| 合计 | 24G 4090 | 预计 16–20GB |

要点：

- reference 用 `eval()` + `torch.no_grad()`；
- 不要对 reference 做 LoRA，也不要 `merge_and_unload` 后当 ref（分布会变）；
- policy 用 LoRA，不要全参。

---

## 4. tokenization 与 logprob

1. 用 Qwen chat template 拼 `user + assistant`；
2. prompt 部分 mask 掉，只对 response token 算 logprob；
3. 右 padding，`labels` 的 prompt/pad 位置设为 `-100`；
4. `logits[:, :-1]` 对 `labels[:, 1:]` 做 `log_softmax + gather`；
5. 分别算 `pi_w`、`pi_l`、`ref_w`、`ref_l`；
6. 检查一条样本：`len(logp) == response_token_count`。

```python
def response_logprob(model, input_ids, labels, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = out[:, :-1, :]
    tgt = labels[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    mask = tgt != -100
    tok_logp = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    # 标准 DPO：sum，不是 mean
    return (tok_logp * mask).sum(-1)
```

```python
pi_w = response_logprob(policy, ...)
pi_l = response_logprob(policy, ...)
with torch.no_grad():
    ref_w = response_logprob(ref, ...)
    ref_l = response_logprob(ref, ...)

logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
loss = -F.logsigmoid(logits).mean()
acc = (logits > 0).float().mean()
```

---

## 5. 训练配置（起点）

| 项 | 值 |
|---|---|
| β | 0.05 |
| batch | 1 × grad accum 16 |
| lr | 5e-6（LoRA 可试 1e-5） |
| scheduler | cosine |
| warmup | 10–20 步 |
| max_len | 2048 |
| smoke | 300 步 |
| full | 800–1200 步 |
| bf16 | 是 |
| grad ckpt | 是 |

---

## 6. 必须记录

- `dpo_loss`；
- `rewards/accuracies = mean(implicit_reward_chosen > implicit_reward_rejected)`；
- `rewards/margins = mean(beta * (Δpolicy - Δref))`；
- `chosen_logps` / `rejected_logps`；
- `ref_chosen_logps` / `ref_rejected_logps`；
- 近似 KL / policy-ref logprob 差；
- 训练集平均 response 长度。

---

## 7. 验收

- [ ] DPO loss 下降，`rewards/accuracies` > 0.6；
- [ ] `rewards/margins` 从 ~0 上升；
- [ ] chosen logp 上升，rejected logp 不暴涨；
- [ ] DPO 前后平均长度有记录；
- [ ] 200 条固定 prompt，base vs DPO judge win rate；
- [ ] 同时报长度分桶 win rate，避免“只靠变长赢”。

---

## 8. 风险与降级

| 风险 | 处理 |
|---|---|
| OOM | seq 1536、bs 1、grad ckpt、LoRA r=16 |
| loss 不降 | lr 1e-5、检查 mask/shift、先过拟合 20 条 |
| 长度暴涨 | 降 β / 长度惩罚 / 长度平衡采样 |
| win rate 全靠长度 | 长度分桶 win rate、LC |
| reference 显存不够 | ref bf16 + `no_grad`；不要同时保留两个 optimizer |

---

## 9. 交付物

- `train_dpo.py`：policy/ref + LoRA + DPO loss + 日志；
- `dpo_smoke.log`：20–50 步 loss/acc/margin/长度；
- `dpo_full.log`：800–1200 步；
- `eval_dpo.py`：200 条固定 prompt，base vs DPO，judge + 长度；
- `dpo_eval.json`：win rate、平均长度、长度分桶 win rate。
