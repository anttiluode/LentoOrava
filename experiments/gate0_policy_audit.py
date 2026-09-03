"""Does the ACTIVE trajectory depend on evidence, or is it a flight plan?"""
from __future__ import annotations

import argparse
import json

import torch

from lentoorava import OrganismConfig, PairCopyConfig, PairCopyDataset, make_variant


def trace_distance_px(a: torch.Tensor, b: torch.Tensor, size: int) -> float:
    rms = torch.sqrt(((a - b) ** 2).sum(dim=-1).mean())
    return float(rms * (size - 1) / 2.0)


def run_trace(model, image, device):
    with torch.no_grad():
        _, trace = model(image[None].to(device), return_trace=True)
    return trace[0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = OrganismConfig(**ckpt["cfg"])
    model = make_variant(ckpt["variant"], cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = PairCopyDataset(1, seed=0, cfg=PairCopyConfig(noise_std=0.0))
    ys = list(ds.y_values)
    y_lo, y_hi = ys[len(ys) // 4], ys[(3 * len(ys)) // 4]
    color_a, color_b = 1, 0

    ref, _ = ds.render(y_lo, color_a, noisy=False)
    y_shift, _ = ds.render(y_hi, color_a, noisy=False)
    color_shift, _ = ds.render(y_lo, color_b, noisy=False)
    blank = torch.zeros_like(ref)

    tr_ref = run_trace(model, ref, args.device)
    result = {
        "variant": ckpt["variant"],
        "reference": {"y": y_lo, "color": color_a},
        "same_color_changed_y_px": trace_distance_px(
            tr_ref, run_trace(model, y_shift, args.device), cfg.image_size
        ),
        "same_y_changed_color_px": trace_distance_px(
            tr_ref, run_trace(model, color_shift, args.device), cfg.image_size
        ),
        "evidence_vs_blank_px": trace_distance_px(
            tr_ref, run_trace(model, blank, args.device), cfg.image_size
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
