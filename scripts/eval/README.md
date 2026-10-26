# 评测器（项目一 / 项目二通用数学评测）

> 2026-10-26：升级为 **v2**（P1-UNIFIED 统一复评口径）。旧版问题：`\boxed{([^{}]*)}` 遇嵌套大括号直接截断（`\boxed{(3, \frac{\pi}{2})}` → `(3, \`），默认 `math_verify.parse` 又丢弃 LaTeX 元组与符号表达式，导致正确答案被判 0；再用"最后一个数字"兜底会放大误判。

## 文件

| 文件 | 作用 |
|---|---|
| `eval_math.py` | 主评测器：vLLM 生成 + 版本化 parser + 逐题 JSONL |
| `summarize_eval.py` | 把多个 result json 汇总成主表（支持 AIME 90 题 / 2024 子集双标签） |
| `paired_delta.py` | 逐题配对 delta + question bootstrap 95% CI |

## 用法

```bash
# 1) 评测单个 checkpoint（P1-UNIFIED 冻结参数）
python scripts/eval/eval_math.py \
  --model /path/to/model-or-merged \
  --data data/eval/math500.jsonl \
  --n 4 --seeds 42,43,44 --max-new-tokens 3072 \
  --prompt-mode legacy --parser v2 \
  --out results/my-math500.json --dump-jsonl results/my-math500.jsonl \
  --run-id my-run

# 2) 汇总（AIME 双标签）
python scripts/eval/summarize_eval.py --dir results/unified/main \
  --subset aime24_2024_ids.json --out results/unified/summary.md

# 3) 配对 delta（vs base）
python scripts/eval/paired_delta.py --base results/base-math500.json \
  --run results/my-math500.json --labels my-run
```

## 协议要点（写论文/报告时必须复述）

- `--prompt-mode legacy`：MATH-500 要求 `\boxed{}`、AIME 要求 `Answer: ...`（与历史 `d2_tools.py` 对齐）；`bare` 为裸题面。
- `--parser v2`：平衡 `\boxed{}` → `Answer:/final answer:` 行 → None；**不做数字兜底**。`legacy` 仅用于复现旧行为（会记录 `parse_mode=number_fallback`，并在 summary 里单列该比例）。
- 每个 sample 一行 JSONL：`run_id/model_id/benchmark/question_id/eval_seed/sample_idx/prompt_mode/parser/prompt_hash/completion/parsed_answer/parse_mode/reference/correct/token_count/truncated/parse_error/evaluator_version`。
- `truncated` 来自 vLLM `finish_reason=="length"`；`token_count` 用模型 tokenizer 计。
- 不要把 `AIME22–24(90)`、`AIME2024(30 子集)`、`AIME2024(旧协议,30)` 混比。
