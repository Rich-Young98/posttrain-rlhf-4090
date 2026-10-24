#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variance-weighted prompt sampler (candidate verl feature / PR).

Idea: GRPO's learning signal is proportional to within-group reward variance.
Prompts with pass-rate p have variance p(1-p); easy (p~1) and impossible (p~0)
prompts produce zero advantage and waste rollout compute. Weight sampling by
4*p*(1-p), or filter to a difficulty band.

This file is intentionally dependency-light so the weighting logic can be unit
tested without torch. The sampler wrapper uses torch.utils.data.WeightedRandomSampler.

Integration sketch (verl v0.8):
  - add config: data.sampler.type=uniform|variance_weighted
                 data.sampler.pass_rate_path=...
                 data.sampler.p_low=0.1, data.sampler.p_high=0.9
  - in verl/trainer/ppo/ray_trainer.py::create_dataloader, when type is
    variance_weighted, wrap the dataset with VarianceWeightedSampler.
  - pass_rate_path: JSON mapping dataset index -> pass-rate (estimated offline
    with the base model, n=8/16).
"""
from __future__ import annotations

import json
from typing import Iterable


def compute_weights(pass_rates: Iterable[float | None], p_low: float = 0.1,
                    p_high: float = 0.9, power: float = 1.0) -> list[float]:
    """Weight = (4p(1-p))^power inside [p_low, p_high], else 0; all-zero -> uniform."""
    rates = list(pass_rates)
    weights: list[float] = []
    for p in rates:
        if p is None:
            weights.append(0.0)
            continue
        p = float(p)
        if p < p_low or p > p_high:
            weights.append(0.0)
        else:
            weights.append(max(0.0, (4.0 * p * (1.0 - p)) ** power))
    if sum(weights) <= 0:
        return [1.0] * len(rates)
    return weights


def load_pass_rates(path: str, size: int, default: float | None = None) -> list[float | None]:
    """Load {index: pass_rate} JSON; missing indices use default."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: list[float | None] = []
    for i in range(size):
        v = raw.get(str(i), raw.get(i, default))
        out.append(None if v is None else float(v))
    return out


class VarianceWeightedSampler:
    """Torch sampler wrapper; import torch lazily so pure tests don't need it."""

    def __init__(self, dataset, pass_rates, p_low: float = 0.1,
                 p_high: float = 0.9, power: float = 1.0, seed: int = 0):
        import torch
        from torch.utils.data import WeightedRandomSampler

        self.dataset = dataset
        weights = compute_weights(pass_rates, p_low, p_high, power)
        self.effective = sum(1 for w in weights if w > 0)
        g = torch.Generator()
        g.manual_seed(seed)
        self._sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=g,
        )

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pass_rate_json", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--p_low", type=float, default=0.1)
    ap.add_argument("--p_high", type=float, default=0.9)
    args = ap.parse_args()
    rates = load_pass_rates(args.pass_rate_json, args.size)
    w = compute_weights(rates, args.p_low, args.p_high)
    print(json.dumps({"n": len(w), "effective": sum(x > 0 for x in w),
                      "mean_weight": sum(w) / max(len(w), 1),
                      "min": min(w), "max": max(w)}, ensure_ascii=False))
