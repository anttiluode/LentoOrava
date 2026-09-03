"""Gate 0A/0B comparison with explicit positive and negative controls."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path("results/gate0a")
    outdir.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "experiments/gate0_baselines.py"], check=True)

    all_runs = {}
    for seed in args.seeds:
        seed_runs = {}
        for variant in ["carrier", "active", "fixed", "transport"]:
            out = outdir / f"{variant}_seed{seed}.pt"
            cmd = [
                sys.executable,
                "train.py",
                "--variant", variant,
                "--epochs", str(args.epochs),
                "--seed", str(seed),
                "--device", args.device,
                "--out", str(out),
            ]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            receipt = json.loads(out.with_suffix(".json").read_text())
            best = min(receipt["history"], key=lambda r: r["val_right_mse"])
            seed_runs[variant] = {
                "best_epoch": best["epoch"],
                "best_right_mse": best["val_right_mse"],
                "final": receipt["history"][-1],
            }
        all_runs[str(seed)] = seed_runs

    path = outdir / "summary.json"
    path.write_text(json.dumps(all_runs, indent=2))
    print(json.dumps(all_runs, indent=2))
    print(path)


if __name__ == "__main__":
    main()
