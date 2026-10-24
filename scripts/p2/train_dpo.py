#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D8：自定义 DPO（不依赖 TRL）。

20 步 smoke：
  MAX_STEPS=20 GRAD_ACCUM=4 MAX_LEN=1024 OUT_DIR=.../dpo-smoke \
  /root/autodl-tmp/venv-verl/bin/python scripts/train_dpo.py

full（示例）：
  MAX_STEPS=800 GRAD_ACCUM=16 MAX_LEN=1536 OUT_DIR=.../dpo-b005 \
  /root/autodl-tmp/venv-verl/bin/python scripts/train_dpo.py
"""
import json
import math
import os
import time

import datasets
import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
DATA = os.environ.get("DPO_DATA", "/root/autodl-tmp/work/data/uf_rm/train.parquet")
OUT = os.environ.get("OUT_DIR", "/root/autodl-tmp/work/runs/dpo-b005")
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "5e-6"))
BETA = float(os.environ.get("BETA", "0.05"))
WARMUP = int(os.environ.get("WARMUP", "5"))
N_TRAIN = int(os.environ.get("N_TRAIN", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
SEED = int(os.environ.get("SEED", "42"))

os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED)

print("[config]", json.dumps({
    "base": BASE, "data": DATA, "out": OUT, "max_len": MAX_LEN,
    "max_steps": MAX_STEPS, "grad_accum": GRAD_ACCUM, "lr": LR,
    "beta": BETA, "warmup": WARMUP, "n_train": N_TRAIN,
}, ensure_ascii=False), flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# policy：base + LoRA
policy = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
policy.config.use_cache = False
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)
policy = get_peft_model(policy, lora_cfg)
policy.print_trainable_parameters()
base_policy = policy.get_base_model()
base_policy.gradient_checkpointing_enable()
base_policy.enable_input_require_grads()
policy.to(device)
policy.train()

# reference：同一 base，冻结，不挂 LoRA
ref = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
ref.config.use_cache = False
ref.to(device).eval()
for p in ref.parameters():
    p.requires_grad = False
torch.cuda.empty_cache()
print("[model] policy + frozen reference loaded", flush=True)


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def build_example(prompt, response):
    """只对 assistant response 算 logprob；prompt + pad 的 label 设为 -100。"""
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": as_text(prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

    full_text = prompt_text + as_text(response) + (tok.eos_token or "")
    enc = tok(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
    )
    ids = enc["input_ids"]
    plen = min(len(prompt_ids), len(ids))
    labels = [-100] * plen + list(ids[plen:])
    if len(labels) != len(ids) or all(x == -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def pad_examples(examples):
    max_len = max(len(x["input_ids"]) for x in examples)
    b = len(examples)
    input_ids = torch.full((b, max_len), tok.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
    return input_ids, attention_mask, labels


def collate(batch):
    chosen = [build_example(x["prompt"], x["chosen"]) for x in batch]
    rejected = [build_example(x["prompt"], x["rejected"]) for x in batch]
    if any(x is None for x in chosen) or any(x is None for x in rejected):
        return None
    c_ids, c_mask, c_labels = pad_examples(chosen)
    r_ids, r_mask, r_labels = pad_examples(rejected)
    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


def response_logprob(model, input_ids, attention_mask, labels):
    """返回每个序列的 response token logprob 之和；不是 token 平均。"""
    out = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = out[:, :-1, :]
    tgt = labels[:, 1:]
    mask = tgt != -100

    flat_logits = logits[mask]          # (N_resp, vocab)
    flat_tgt = tgt[mask]                # (N_resp,)
    logp = F.log_softmax(flat_logits.float(), dim=-1)
    tok_logp = logp.gather(-1, flat_tgt.unsqueeze(-1)).squeeze(-1)

    bsz = input_ids.size(0)
    batch_idx = torch.arange(bsz, device=input_ids.device).unsqueeze(1).expand_as(mask)[mask]
    sums = torch.zeros(bsz, device=input_ids.device)
    sums = sums.index_add(0, batch_idx, tok_logp)
    return sums


ds = datasets.load_dataset("parquet", data_files=DATA, split="train")
ds = ds.shuffle(seed=SEED)
if N_TRAIN > 0:
    ds = ds.select(range(min(N_TRAIN, len(ds))))
print("[data]", len(ds), ds.column_names, flush=True)

loader = DataLoader(
    ds,
    batch_size=1,
    shuffle=True,
    collate_fn=collate,
    num_workers=NUM_WORKERS,
    drop_last=True,
)

trainable = [p for p in policy.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)


def lr_scale(step):
    """step 从 1 开始。"""
    if step <= WARMUP:
        return step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


data_iter = iter(loader)
start = time.time()
for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    step_acc = 0.0
    step_margin = 0.0
    step_pi_w = step_pi_l = step_ref_w = step_ref_l = 0.0
    step_clen = step_rlen = 0.0
    used = 0

    for micro in range(GRAD_ACCUM):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        if batch is None:
            continue

        c_ids = batch["chosen_input_ids"].to(device)
        c_mask = batch["chosen_attention_mask"].to(device)
        c_labels = batch["chosen_labels"].to(device)
        r_ids = batch["rejected_input_ids"].to(device)
        r_mask = batch["rejected_attention_mask"].to(device)
        r_labels = batch["rejected_labels"].to(device)

        pi_w = response_logprob(policy, c_ids, c_mask, c_labels)
        pi_l = response_logprob(policy, r_ids, r_mask, r_labels)
        with torch.no_grad():
            ref_w = response_logprob(ref, c_ids, c_mask, c_labels)
            ref_l = response_logprob(ref, r_ids, r_mask, r_labels)

        dpo_logits = BETA * ((pi_w - ref_w) - (pi_l - ref_l))
        loss = -F.logsigmoid(dpo_logits).mean()

        (loss / GRAD_ACCUM).backward()
        used += 1

        step_loss += loss.item()
        step_acc += (dpo_logits > 0).float().mean().item()
        step_margin += dpo_logits.mean().item()
        step_pi_w += pi_w.mean().item()
        step_pi_l += pi_l.mean().item()
        step_ref_w += ref_w.mean().item()
        step_ref_l += ref_l.mean().item()
        step_clen += (c_labels != -100).sum(1).float().mean().item()
        step_rlen += (r_labels != -100).sum(1).float().mean().item()

        if step == 1 and micro == 0:
            print("[debug] chosen_len", int((c_labels != -100).sum()),
                  "rejected_len", int((r_labels != -100).sum()),
                  "chosen_ids_len", int(c_ids.size(1)),
                  "rejected_ids_len", int(r_ids.size(1)), flush=True)

    if used == 0:
        print("[warn] step", step, "all batches invalid, skip", flush=True)
        continue

    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    cur_lr = LR * lr_scale(step)
    for g in optimizer.param_groups:
        g["lr"] = cur_lr
    optimizer.step()

    n = used
    msg = {
        "step": step,
        "loss": round(step_loss / n, 4),
        "acc": round(step_acc / n, 4),
        "margin": round(step_margin / n, 4),
        "pi_w": round(step_pi_w / n, 3),
        "pi_l": round(step_pi_l / n, 3),
        "ref_w": round(step_ref_w / n, 3),
        "ref_l": round(step_ref_l / n, 3),
        "len_w": round(step_clen / n, 1),
        "len_l": round(step_rlen / n, 1),
        "lr": f"{cur_lr:.2e}",
        "sec": round(time.time() - start, 1),
    }
    print(json.dumps(msg, ensure_ascii=False), flush=True)

final_dir = os.path.join(OUT, "final")
policy.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
print("[done] adapter saved to", final_dir, flush=True)
