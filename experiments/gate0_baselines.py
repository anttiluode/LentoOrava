"""Exact boring baselines for Gate 0's hidden-right metric."""
from __future__ import annotations

import argparse
import json

import torch

from lentoorava import PairCopyConfig, PairCopyDataset


def mse_right(pred: torch.Tensor, target: torch.Tensor) -> float:
    mid = target.shape[-1] // 2
    return ((pred[..., mid:] - target[..., mid:]) ** 2).mean().item()


def build_attackers(ds: PairCopyDataset):
    by_pair = {}
    for y in ds.y_values:
        for c in range(ds.n_colors):
            observed, target = ds.render(y, c, noisy=False)
            by_pair[(y, c)] = (observed, target)

    all_targets = torch.stack([t for _, t in by_pair.values()])
    mean_target = all_targets.mean(dim=0)
    mean_by_y = {
        y: torch.stack([by_pair[(y, c)][1] for c in range(ds.n_colors)]).mean(dim=0)
        for y in ds.y_values
    }
    mean_by_color = {
        c: torch.stack([by_pair[(y, c)][1] for y in ds.y_values]).mean(dim=0)
        for c in range(ds.n_colors)
    }
    return by_pair, mean_target, mean_by_y, mean_by_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=50_000)
    args = ap.parse_args()

    ds = PairCopyDataset(args.val_size, seed=args.seed, cfg=PairCopyConfig(noise_std=0.02))
    by_pair, mean_target, mean_by_y, mean_by_color = build_attackers(ds)
    sums = {k: 0.0 for k in ["zeros", "identity", "dataset_mean", "knows_y", "knows_color", "oracle"]}

    for i in range(len(ds)):
        observed, target, meta = ds[i]
        y = int(meta["y"])
        c = int(meta["color"])
        preds = {
            "zeros": torch.zeros_like(target),
            "identity": observed,
            "dataset_mean": mean_target,
            "knows_y": mean_by_y[y],
            "knows_color": mean_by_color[c],
            "oracle": by_pair[(y, c)][1],
        }
        for name, pred in preds.items():
            sums[name] += mse_right(pred, target)

    result = {name: value / len(ds) for name, value in sums.items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
