"""Run the first three-arm falsification test.

ACTIVE    local read + private state + movable local write
FIXED     same agents, same parameter budget, addresses cannot move
TRANSPORT no agents; only the shared local transport field
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path("results/gate0")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for variant in ["active", "fixed", "transport"]:
        out = outdir / f"{variant}_seed{args.seed}.pt"
        cmd = [
            sys.executable,
            "train.py",
            "--variant",
            variant,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--out",
            str(out),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        receipt = json.loads(out.with_suffix(".json").read_text())
        summary[variant] = receipt["history"][-1]

    (outdir / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
