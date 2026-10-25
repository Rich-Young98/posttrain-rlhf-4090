#!/usr/bin/env python3
"""Merge a verl/PEFT LoRA adapter into the base model and verify weights changed.

Why this exists
---------------
`verl.model_merger` for LoRA checkpoints exports the *base* weights plus a
separate `lora_adapter/` directory; it does NOT multiply the adapter into the
base. Evaluating the merger output directly gives exactly the base scores
(this project hit that bug). This script does the PEFT merge and a sanity check.

Usage
-----
python scripts/p1/merge_lora_peft.py \
  --base /root/autodl-tmp/models/Qwen2.5-Math-1.5B-Instruct \
  --adapter /root/autodl-tmp/work/merged/p1-r1-300-lora \
  --out /root/autodl-tmp/work/merged/p1-r1-300-hf
"""
from __future__ import annotations

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base HF model directory")
    ap.add_argument("--adapter", required=True, help="PEFT LoRA adapter directory")
    ap.add_argument("--out", required=True, help="output merged HF model directory")
    ap.add_argument(
        "--check-modules",
        nargs="+",
        default=["q_proj", "o_proj", "down_proj"],
        help="substrings of parameters used for the max-diff check",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[load] base={args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    ref = {k: v.detach().clone() for k, v in base.state_dict().items()}

    print(f"[load] adapter={args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    after = model.state_dict()
    print("[check] max |merged - base| for selected modules:")
    changed = 0
    for pattern in args.check_modules:
        diffs = [
            (k, float((after[k].float() - ref[k].float()).abs().max()))
            for k in after
            if pattern in k and k in ref and after[k].shape == ref[k].shape
        ]
        if not diffs:
            print(f"  {pattern}: (no matching parameter)")
            continue
        name, diff = max(diffs, key=lambda x: x[1])
        print(f"  {pattern}: maxdiff={diff:.6g} ({name})")
        if diff > 0:
            changed += 1

    if changed == 0:
        print(
            "[error] no weight changed. Did you point --adapter at a base copy "
            "instead of a lora_adapter directory?",
            file=sys.stderr,
        )
        return 1

    print(f"[save] {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    print("[done] verify eval scores differ from base before trusting them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
