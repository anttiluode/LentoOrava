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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def losses(pred: torch.Tensor, target: torch.Tensor):
    full = ((pred - target) ** 2).mean()
    mid = pred.shape[-1] // 2
    right = ((pred[..., :, mid:] - target[..., :, mid:]) ** 2).mean()
    return full + 4.0 * right, full, right


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(3, device=device)
    count = 0
    for observed, target, _ in loader:
        observed, target = observed.to(device), target.to(device)
        pred = model(observed)
        vals = losses(pred, target)
        totals += torch.tensor([v.item() for v in vals], device=device) * observed.shape[0]
        count += observed.shape[0]
    return (totals / count).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["active", "fixed", "transport"], default="active")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-size", type=int, default=4096)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = OrganismConfig(steps=args.steps, n_agents=args.agents, channels=args.channels)
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
        for observed, target, _ in train_loader:
            observed, target = observed.to(device), target.to(device)
            pred = model(observed)
            loss, _, _ = losses(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * observed.shape[0]
            seen += observed.shape[0]

        val_total, val_full, val_right = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "val_loss": val_total,
            "val_full_mse": val_full,
            "val_right_mse": val_right,
        }
        history.append(row)
        print(json.dumps(row))

    out = Path(args.out or f"results/{args.variant}_seed{args.seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": vars(cfg),
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
