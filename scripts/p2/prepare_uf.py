#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D6：清洗 UltraFeedback，产出 RM 训练/留出 parquet 和统计。

远端路径固定为：
  raw:  /root/autodl-tmp/work/data/raw/uf
  out:  /root/autodl-tmp/work/data/uf_rm
用法：
  /root/autodl-tmp/venv-verl/bin/python /root/autodl-tmp/work/scripts/prepare_uf.py
"""
import glob
import hashlib
import json
import os

import datasets
import numpy as np
from transformers import AutoTokenizer

RAW = "/root/autodl-tmp/work/data/raw/uf"
OUT = "/root/autodl-tmp/work/data/uf_rm"
BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
N_TRAIN = 30000

os.makedirs(OUT, exist_ok=True)


def as_text(x):
    """chosen/rejected 一般是 str；兼容 list[str] / list[dict] / dict。"""
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def norm(s):
    return " ".join(as_text(s).split())


def load_raw():
    files = sorted(glob.glob(os.path.join(RAW, "**", "*.parquet"), recursive=True))
    if files:
        print(f"[load] 本地 parquet: {len(files)} 个文件")
        return datasets.load_dataset("parquet", data_files=files, split="train")
    print("[load] 本地没找到 parquet，退回 hub 下载（需要网络）")
    return datasets.load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned", split="train"
    )


ds = load_raw()
print("[raw]", ds)
print("[cols]", ds.column_names)

# ModelScope 的 mlabonne 镜像用 question 列名，统一成 prompt
if "prompt" not in ds.column_names and "question" in ds.column_names:
    ds = ds.rename_column("question", "prompt")
    print("[cols] question -> prompt")

for col in ("prompt", "chosen", "rejected"):
    if col not in ds.column_names:
        raise SystemExit(f"缺列 {col}，实际列：{ds.column_names}")

before = len(ds)
ds = ds.filter(
    lambda x: len(norm(x["chosen"])) > 0
    and len(norm(x["rejected"])) > 0
    and norm(x["chosen"]) != norm(x["rejected"])
)
print(f"[clean] {before} -> {len(ds)}（去掉空回答和 chosen==rejected）")

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def n_tokens(s):
    return len(tok(as_text(s), add_special_tokens=False)["input_ids"])


# 全量字符长度：便宜，用来看“chosen 更长”的比例
chosen_longer_char = sum(
    len(as_text(x["chosen"])) > len(as_text(x["rejected"])) for x in ds
)
chosen_longer_char_ratio = chosen_longer_char / max(len(ds), 1)

# 抽样 3000 条做 token 长度分布，避免 6 万条全量 tokenize 太慢
sample_n = min(3000, len(ds))
sample = ds.shuffle(seed=0).select(range(sample_n))
c_lens, r_lens = [], []
for x in sample:
    c_lens.append(n_tokens(x["chosen"]))
    r_lens.append(n_tokens(x["rejected"]))
c_lens = np.asarray(c_lens)
r_lens = np.asarray(r_lens)
diff = c_lens - r_lens

stats = {
    "raw_rows": before,
    "clean_rows": len(ds),
    "chosen_longer_char_ratio": round(float(chosen_longer_char_ratio), 4),
    "sample_n": sample_n,
    "chosen_tokens_p50_p90_p99": [int(v) for v in np.percentile(c_lens, [50, 90, 99])],
    "rejected_tokens_p50_p90_p99": [int(v) for v in np.percentile(r_lens, [50, 90, 99])],
    "chosen_minus_rejected_tokens_mean": round(float(diff.mean()), 2),
    "chosen_longer_token_ratio_sample": round(float((diff > 0).mean()), 4),
    "abs_diff_gt_1000_ratio_sample": round(float((np.abs(diff) > 1000).mean()), 4),
}
print("[stats]", json.dumps(stats, ensure_ascii=False, indent=2))

with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 固定 seed，并且按 prompt 分桶：同一 prompt 的多个 pair 不能同时进 train/holdout，
# 否则 UF held-out 会被“同题泄漏”抬高。
def prompt_bucket(p):
    h = hashlib.md5(norm(p).encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % 100


holdout = ds.filter(lambda x: prompt_bucket(x["prompt"]) < 2)
train_pool = ds.filter(lambda x: prompt_bucket(x["prompt"]) >= 2)
train = train_pool.shuffle(seed=42).select(range(min(N_TRAIN, len(train_pool))))

train_path = os.path.join(OUT, "train.parquet")
holdout_path = os.path.join(OUT, "holdout.parquet")
train.to_parquet(train_path)
holdout.to_parquet(holdout_path)

print(f"[split] holdout={len(holdout)}  train_pool={len(train_pool)}  train={len(train)}")
print(f"[done] train={len(train)} -> {train_path}")
print(f"[done] holdout={len(holdout)} -> {holdout_path}")
print(f"[done] stats -> {os.path.join(OUT, 'stats.json')}")
