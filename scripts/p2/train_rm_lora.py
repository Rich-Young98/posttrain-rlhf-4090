#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D6：Qwen2.5-1.5B-Instruct + LoRA 奖励模型（BT loss）。

24G 单卡可跑：
  MAX_STEPS=100  -> smoke，看 loss 是否下降
  MAX_STEPS=1875 -> 约 1 epoch（30k pair / 有效 batch 16）

用法：
  MAX_STEPS=100 /root/autodl-tmp/venv-verl/bin/python \
    /root/autodl-tmp/work/scripts/train_rm_lora.py
"""
import os

import datasets
import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE = os.environ.get("BASE_MODEL", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
DATA = os.environ.get("DATA", "/root/autodl-tmp/work/data/uf_rm/train.parquet")
OUT = os.environ.get("OUT_DIR", "/root/autodl-tmp/work/runs/rm-lora")
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
LR = float(os.environ.get("LR", "1e-4"))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1") == "1"

os.makedirs(OUT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tok.pad_token_id
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # 明确列 attention + MLP，不要 all-linear（会把 score 也包进去）
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # score 是随机初始化的分类头，必须存下来，否则重载后等于没训练
    modules_to_save=["score"],
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

if GRAD_CKPT:
    # 显式拿 base model，避免 PEFT 版本差异导致方法转发不到
    base_model = model.get_base_model()
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    print("[mem] gradient checkpointing ON")

ds = datasets.load_dataset("parquet", data_files=DATA, split="train")
print("[data]", ds)


def as_text(x):
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return " ".join(str(m.get("content", "")) for m in x)
        return " ".join(str(t) for t in x)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def render(prompt, response):
    msgs = [
        {"role": "user", "content": as_text(prompt)},
        {"role": "assistant", "content": as_text(response)},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def collate(batch):
    chosen = tok(
        [render(x["prompt"], x["chosen"]) for x in batch],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    rejected = tok(
        [render(x["prompt"], x["rejected"]) for x in batch],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }


class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        ).logits.view(-1)
        r = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        ).logits.view(-1)
        loss = -F.logsigmoid(c - r).mean()
        # 顺带记录 pair accuracy（不是最终指标，只用于看是否在学）
        self._last_acc = (c > r).float().mean().item()
        return (loss, {"loss": loss}) if return_outputs else loss

    def log(self, logs, start_time=None):
        if hasattr(self, "_last_acc"):
            logs["train/pair_acc"] = self._last_acc
        super().log(logs, start_time=start_time)


args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=5,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    seed=42,
    data_seed=42,
)

trainer = RMTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate,
)
trainer.train()

final_dir = os.path.join(OUT, "final")
trainer.save_model(final_dir)
tok.save_pretrained(final_dir)
print(f"[done] adapter + score 已保存到 {final_dir}")
