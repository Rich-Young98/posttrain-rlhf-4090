#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 UF train 按 prompt 哈希切成 A/B 两份，用于 proxy RM / gold RM。

A = bucket 0..49, B = bucket 50..99；同一 prompt 只进一边。
产物：
  /root/autodl-tmp/work/data/uf_rm/train_A.parquet
  /root/autodl-tmp/work/data/uf_rm/train_B.parquet
  /root/autodl-tmp/work/data/uf_rm/split_ab_stats.json
"""
import hashlib
import json
import os

import datasets

SRC = "/root/autodl-tmp/work/data/uf_rm/train.parquet"
OUT_DIR = "/root/autodl-tmp/work/data/uf_rm"


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def norm(s):
    return " ".join(as_text(s).split())


def bucket(p):
    h = hashlib.md5(norm(p).encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % 100


ds = datasets.load_dataset("parquet", data_files=SRC, split="train")
print("[raw]", len(ds), ds.column_names)

a = ds.filter(lambda x: bucket(x["prompt"]) < 50)
b = ds.filter(lambda x: bucket(x["prompt"]) >= 50)

pa = {norm(x["prompt"]) for x in a}
pb = {norm(x["prompt"]) for x in b}
print("[split] A", len(a), "B", len(b), "prompt_overlap", len(pa & pb))


def stats(d):
    n = len(d)
    longer = sum(len(as_text(x["chosen"])) > len(as_text(x["rejected"])) for x in d)
    return {"n": n, "chosen_longer_char_ratio": round(longer / max(n, 1), 4)}


st = {
    "source": SRC,
    "raw": len(ds),
    "A": stats(a),
    "B": stats(b),
    "prompt_overlap": len(pa & pb),
}
print("[stats]", json.dumps(st, ensure_ascii=False, indent=2))

a_path = os.path.join(OUT_DIR, "train_A.parquet")
b_path = os.path.join(OUT_DIR, "train_B.parquet")
a.to_parquet(a_path)
b.to_parquet(b_path)
with open(os.path.join(OUT_DIR, "split_ab_stats.json"), "w", encoding="utf-8") as f:
    json.dump(st, f, ensure_ascii=False, indent=2)

print("[done]", a_path, b_path)
