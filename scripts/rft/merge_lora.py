#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge a LoRA adapter into the base model for vLLM evaluation.

Usage:
  python scripts/rft/merge_lora.py \
    --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
    --adapter /root/autodl-tmp/work/runs/rft-lora \
    --out /root/autodl-tmp/work/runs/rft-merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[done] merged -> {args.out}")


if __name__ == "__main__":
    main()
