#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D10：best-of-N over-optimization（proxy RM 选，gold RM 打分）。

流程：
1) base policy 对每个 prompt 采 n_samples 条；
2) proxy RM 给所有样本打分；
3) 对 N in ns，选 proxy 最高的前 N 条（nested）；
4) gold RM 只给被选中的回答打分；
5) 输出 N vs proxy/gold/length 曲线。

用法（24G，200 prompt × 64 samples）：
  /root/autodl-tmp/venv-verl/bin/python scripts/best_of_n.py \
    --n_prompts 200 --n_samples 64 --ns 1,2,4,8,16,32,64 \
    --out_dir /root/autodl-tmp/work/results/bon

如果 vLLM 起不来，先用 --n_prompts 50 --n_samples 16 验证。
"""
import argparse
import gc
import json
import os
import random

import datasets
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
HOLDOUT = "/root/autodl-tmp/work/data/uf_rm/holdout.parquet"


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def load_prompts(n, seed):
    ds = datasets.load_dataset("parquet", data_files=HOLDOUT, split="train")
    seen, prompts = set(), []
    for x in ds:
        p = " ".join(as_text(x["prompt"]).split())
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
    random.Random(seed).shuffle(prompts)
    return prompts[: min(n, len(prompts))]


def load_rm(adapter, device):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=1, torch_dtype=torch.bfloat16
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).to(device).eval()
    return tok, model


@torch.no_grad()
def score_pairs(tok, model, pairs, device, bs=8, max_len=2048):
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i : i + bs]
        texts = []
        for prompt, resp in chunk:
            texts.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": resp}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        enc = tok(
            texts, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.view(-1).float().cpu().tolist()
        out.extend(logits)
        if (i // bs) % 50 == 0:
            print(f"[score] {min(i + bs, len(pairs))}/{len(pairs)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_adapter", default="/root/autodl-tmp/work/runs/rm-proxy/final")
    ap.add_argument("--gold_adapter", default="/root/autodl-tmp/work/runs/rm-gold/final")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--ns", default="1,2,4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--out_dir", default="/root/autodl-tmp/work/results/bon")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ns = [n for n in ns if n <= args.n_samples]
    prompts = load_prompts(args.n_prompts, args.seed)
    print(f"[data] prompts={len(prompts)} n_samples={args.n_samples} ns={ns}", flush=True)

    # 1) vLLM 生成；base policy，不挂 DPO/RM
    from vllm import LLM, SamplingParams

    vllm_gpu_mem = float(os.environ.get("VLLM_GPU_MEM", "0.80"))
    vllm_max_len = int(os.environ.get("VLLM_MAX_LEN", "2048"))
    llm = LLM(
        model=BASE,
        dtype="bfloat16",
        max_model_len=vllm_max_len,
        gpu_memory_utilization=vllm_gpu_mem,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    chat_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    print("[vllm] generating ...", flush=True)
    outputs = llm.generate(chat_prompts, sampling)
    print("[vllm] done", flush=True)

    samples = []
    for prompt, out in zip(prompts, outputs):
        resps = [o.text.strip() for o in out.outputs]
        samples.append({"prompt": prompt, "responses": resps})
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 2) proxy 给所有样本打分
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy_tok, proxy = load_rm(args.proxy_adapter, device)
    print("[proxy] scoring", flush=True)
    for i, s in enumerate(samples):
        pairs = [(s["prompt"], r) for r in s["responses"]]
        s["proxy_scores"] = score_pairs(proxy_tok, proxy, pairs, device)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[proxy] prompt {i + 1}/{len(samples)}", flush=True)
    del proxy
    gc.collect()
    torch.cuda.empty_cache()

    # 3) 每个 N 选 proxy 最高的前 N 条；gold 只打分被选中的
    gold_tok, gold = load_rm(args.gold_adapter, device)
    print("[gold] scoring selected", flush=True)
    curve = []
    selected_records = []
    for N in ns:
        pairs, meta = [], []
        for s in samples:
            k = min(N, len(s["responses"]))
            idx = max(range(k), key=lambda j: s["proxy_scores"][j])
            resp = s["responses"][idx]
            pscore = s["proxy_scores"][idx]
            pairs.append((s["prompt"], resp))
            meta.append((s["prompt"], resp, pscore, idx))
        gold_scores = score_pairs(gold_tok, gold, pairs, device)
        for (prompt, resp, pscore, idx), gscore in zip(meta, gold_scores):
            ntok = len(gold_tok(resp, add_special_tokens=False)["input_ids"])
            selected_records.append({
                "N": N, "prompt": prompt, "response": resp,
                "proxy": pscore, "gold": gscore, "len": ntok, "idx": idx,
            })
        curve.append({
            "N": N,
            "n": len(meta),
            "proxy_mean": round(sum(x[2] for x in meta) / len(meta), 4),
            "gold_mean": round(sum(gold_scores) / len(gold_scores), 4),
            "len_mean": round(
                sum(len(gold_tok(x[1], add_special_tokens=False)["input_ids"]) for x in meta) / len(meta), 1
            ),
        })
        print("[curve]", json.dumps(curve[-1], ensure_ascii=False), flush=True)

    summary = {
        "n_prompts": len(prompts),
        "n_samples": args.n_samples,
        "ns": ns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "proxy_adapter": args.proxy_adapter,
        "gold_adapter": args.gold_adapter,
        "curve": curve,
    }
    with open(os.path.join(args.out_dir, "bon_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "bon_selected.jsonl"), "w", encoding="utf-8") as f:
        for r in selected_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "bon_samples.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("[done]", os.path.join(args.out_dir, "bon_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
