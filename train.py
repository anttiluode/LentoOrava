from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lentoorava import OrganismConfig, PairCopyDataset, make_variant
from lentoorava.losses import LossConfig, organism_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_cfg, progress: float):
    model.eval()
    sums = {"loss": 0.0, "full_mse": 0.0, "right_mse": 0.0, "pyramid": 0.0, "mass": 0.0}
    count = 0
    for observed, target, _ in loader:
        observed, target = observed.to(device), target.to(device)
        pred = model(observed)
        total, metrics = organism_loss(pred, target, progress=progress, cfg=loss_cfg)
        b = observed.shape[0]
        sums["loss"] += total.item() * b
        for key in ["full_mse", "right_mse", "pyramid", "mass"]:
            sums[key] += metrics[key].item() * b
        count += b
    return {k: v / count for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["active", "carrier", "fixed", "transport"], default="active")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-size", type=int, default=4096)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pyramid-gain", type=float, default=0.05)
    ap.add_argument("--mass-gain", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = OrganismConfig(steps=args.steps, n_agents=args.agents, channels=args.channels)
    loss_cfg = LossConfig(pyramid_gain=args.pyramid_gain, mass_gain=args.mass_gain)
    model = make_variant(args.variant, cfg).to(device)

    train_ds = PairCopyDataset(args.train_size, seed=args.seed * 100_000 + 1)
    val_ds = PairCopyDataset(args.val_size, seed=args.seed * 100_000 + 50_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        for observed, target, _ in train_loader:
            observed, target = observed.to(device), target.to(device)
            pred = model(observed)
            loss, _ = organism_loss(pred, target, progress=progress, cfg=loss_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * observed.shape[0]
            seen += observed.shape[0]

        val = evaluate(model, val_loader, device, loss_cfg, progress)
        row = {
            "epoch": epoch,
            "progress": progress,
            "train_loss": running / seen,
            "val_loss": val["loss"],
            "val_full_mse": val["full_mse"],
            "val_right_mse": val["right_mse"],
            "val_pyramid": val["pyramid"],
            "val_mass": val["mass"],
        }
        history.append(row)
        print(json.dumps(row))

    out = Path(args.out or f"results/{args.variant}_seed{args.seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": vars(cfg),
            "loss_cfg": vars(loss_cfg),
            "variant": args.variant,
            "history": history,
        },
        out,
    )
    out.with_suffix(".json").write_text(
        json.dumps({"variant": args.variant, "history": history}, indent=2)
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
