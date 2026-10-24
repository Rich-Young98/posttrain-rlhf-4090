#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RFT step 2: LoRA SFT on filtered (prompt, response) pairs.

Only response tokens contribute to the loss; prompt tokens are masked (-100).

Usage:
  python scripts/rft/train_sft_lora.py \
    --data /root/autodl-tmp/work/data/rft_math/train.jsonl \
    --out /root/autodl-tmp/work/runs/rft-lora \
    --epochs 1 --lr 1e-4 --max-len 2048
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        prompt_text = self.tok.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True)
        response = ex["response"].strip() + self.tok.eos_token
        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        r_ids = self.tok(response, add_special_tokens=False).input_ids
        input_ids = (p_ids + r_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + r_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad = self.tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for x in batch:
            n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad] * n)
            labels.append(x["labels"] + [-100] * n)
            attn.append(x["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    print(f"[data] {len(rows)} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else "none",
        run_name=os.path.basename(args.out),
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=SFTDataset(rows, tok, args.max_len),
                      data_collator=Collator(tok))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
