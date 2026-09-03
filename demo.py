from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lentoorava import OrganismConfig, PairCopyDataset, make_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--index", type=int, default=7)
    ap.add_argument("--out", default="results/demo.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = OrganismConfig(**ckpt["cfg"])
    model = make_variant(ckpt["variant"], cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    observed, target, _ = PairCopyDataset(64, seed=50_000)[args.index]
    with torch.no_grad():
        pred, trace = model(observed[None].to(args.device), return_trace=True)
    pred = pred[0].cpu()
    trace = trace[0].cpu()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(observed.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title("observed")
    axes[1].imshow(target.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("target")
    axes[2].imshow(pred.permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title("organism")
    axes[3].imshow(observed.permute(1, 2, 0).clamp(0, 1))
    for i in range(trace.shape[1]):
        x = (trace[:, i, 0] + 1) * 0.5 * (cfg.image_size - 1)
        y = (trace[:, i, 1] + 1) * 0.5 * (cfg.image_size - 1)
        axes[3].plot(x, y, linewidth=0.8, alpha=0.7)
    axes[3].set_title("agent paths")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
