# Independent judge（P0）

项目二的 DPO / best-of-N 结论原来只有 **RM proxy**（UltraFeedback 训练的 1.5B RM），而所谓 `gold RM` 也是同模型、同分布、同长度偏置。面试官会问："你的 gold 和 proxy 都是 UF RM，凭什么说过优化/没过优化？"

这个目录用**独立 judge**（API 强模型或本地 7B）做 pairwise 比较，交换 A/B 位置、多票、报长度分桶，补上这块证据。

## 1. 安装

```bash
pip install openai
# local judge 需要：
pip install transformers accelerate bitsandbytes
```

## 2. API judge（推荐，最便宜）

```bash
export JUDGE_API_KEY=sk-...
export JUDGE_BASE_URL=https://api.deepseek.com/v1   # OpenAI: https://api.openai.com/v1
export JUDGE_MODEL=deepseek-chat                    # OpenAI: gpt-4o-mini / gpt-4.1-mini
```

> 任何 OpenAI-compatible 服务都可以（DeepSeek、硅基流动、OpenRouter、vLLM server）。

## 3. DPO：生成 base vs DPO 的配对输出

```bash
BASE=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
WORK=/root/autodl-tmp/work

# DPO 800
python scripts/judge/generate_pair_outputs.py \
  --base_model $BASE \
  --adapter $WORK/runs/dpo-b005/final \
  --prompts_parquet $WORK/data/uf_rm/holdout.parquet \
  --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
  --out $WORK/results/dpo800_pairs.jsonl --label_a base --label_b dpo800

# DPO 300（如果有独立 adapter）
python scripts/judge/generate_pair_outputs.py \
  --base_model $BASE \
  --adapter $WORK/runs/dpo-b005-300/final \
  --prompts_parquet $WORK/data/uf_rm/holdout.parquet \
  --n 200 --max_new_tokens 256 --temperature 0.7 --top_p 0.9 \
  --out $WORK/results/dpo300_pairs.jsonl --label_a base --label_b dpo300
```

## 4. 跑 judge

```bash
python scripts/judge/judge_pairwise.py \
  --pairs $WORK/results/dpo800_pairs.jsonl \
  --out $WORK/results/dpo800_judge.json \
  --backend api --votes 3
```

输出：

- `summary.win_rate_dpo800_excl_ties`：独立 judge 下 DPO 的胜率（不含 ties）；
- `summary.win_rate_dpo800_incl_ties`：ties 算 0.5；
- `summary.length_bins`：base 更长 / 长度相近 / DPO 更长 三桶的胜率；
- `pairs[].votes`：每次 A/B 顺序的原始判断，可审计。

## 5. best-of-N：N=1 vs N=64/128

```bash
# 1) 从 selected 文件构造配对
python scripts/judge/judge_bon_pairs.py \
  --selected "$WORK/results/bon128_selected .jsonl" \
  --ns 64 128 \
  --out $WORK/results/bon_pairs.jsonl

# 2) 跑 judge
python scripts/judge/judge_pairwise.py \
  --pairs $WORK/results/bon_pairs.jsonl \
  --out $WORK/results/bon_judge.json \
  --backend api --votes 3
```

## 6. 本地 7B judge（没有 API key 时）

```bash
python scripts/judge/judge_pairwise.py \
  --pairs $WORK/results/dpo800_pairs.jsonl \
  --out $WORK/results/dpo800_judge_local.json \
  --backend local --model Qwen/Qwen2.5-7B-Instruct --load-4bit --votes 2
```

24G 上 7B 4-bit 可以跑，但会慢；建议先生成配对（1.5B），再单独加载 judge。

## 7. 怎么解读

| 结果 | 含义 |
|---|---|
| DPO judge 胜率明显低于 RM proxy（67.5%） | 证实 RM proxy 高估；DPO mixed/负结果更硬 |
| DPO judge 胜率接近 67.5%，且 similar 桶也高 | DPO 有真实提升（但仍要报长度） |
| BoN N=64/128 judge 胜率不再涨/下降 | 独立 judge 下复现过优化拐点 |
| BoN judge 继续涨 | 支持"未过优化"，且不是同偏置 RM 造成的 |

**红线**：judge 不是 ground truth；要报 votes/ties、位置交换、长度分桶；不要只报一个总胜率。

## 8. 成本

- 200 对 × 3 票 ≈ 600 次 API 调用；DeepSeek 约 ¥5–15；gpt-4o-mini 约 $1–3。
- 本地 7B：2–4 GPU·h（生成+judge）。

## 9. 如果 judge 全是 tie（先看 reason）

2026-10-25 遇到过：200/200 tie，打开 `pairs[0].votes[].reason` 发现是
`ERROR: Error code: 401 - Authentication Fails ...`。
**这不是 judge 认为打平，是 API key 无效。** 脚本现在会在 401/403 时直接退出，
不会再写一份"全 tie"的假结果。

排查：

```bash
echo "${JUDGE_API_KEY:0:8}"                 # 确认不是空/占位符
curl -s https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer $JUDGE_API_KEY" | head
```

- DeepSeek：`JUDGE_BASE_URL=https://api.deepseek.com/v1`、`JUDGE_MODEL=deepseek-chat`；
- OpenAI：`JUDGE_BASE_URL=https://api.openai.com/v1`、`JUDGE_MODEL=gpt-4o-mini`；
- 不要把 OpenAI key 配 DeepSeek base，反之亦然。

如果暂时没有可用 key，用本地 7B：

```bash
python scripts/judge/judge_pairwise.py \
  --pairs "$WORK/results/dpo300_pairs.jsonl" \
  --out "$WORK/results/dpo300_judge_local.json" \
  --backend local --model Qwen/Qwen2.5-7B-Instruct --load-4bit --votes 2
```

如果 `*_pairs.jsonl` 没有保留，从旧的 judge json 恢复：

```bash
python scripts/judge/extract_pairs.py \
  --judge_json p2/dpo300_judge.json \
  --out "$WORK/results/dpo300_pairs.jsonl"
```

## 10. best-of-N 文件找不到

```bash
find /root/autodl-tmp/work -iname "*bon128*selected*"
# 找到后：
BON_SEL="/path/to/bon128_selected .jsonl" bash scripts/judge/run_judge_bon.sh
```
