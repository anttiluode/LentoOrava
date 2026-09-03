from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    scales: tuple[float, ...] = (0.0, 1.5, 3.0, 6.0)
    pyramid_gain: float = 0.05
    mass_gain: float = 0.15
    eps: float = 1e-6


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    q = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (q / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(x, kh, padding=(0, radius), groups=c)
    x = F.conv2d(x, kv, padding=(radius, 0), groups=c)
    return x


def curriculum_weights(progress: float, scales: tuple[float, ...] = (0.0, 1.5, 3.0, 6.0)) -> tuple[float, ...]:
    if tuple(scales) != (0.0, 1.5, 3.0, 6.0):
        p = float(min(max(progress, 0.0), 1.0))
        n = len(scales)
        center = (1.0 - p) * (n - 1)
        return tuple(
            math.exp(-0.5 * ((i - center) / max(0.8, n / 4)) ** 2)
            for i in range(n)
        )
    p = float(min(max(progress, 0.0), 1.0))
    if p < 1.0 / 3.0:
        return (0.0, 0.10, 0.70, 1.00)
    if p < 2.0 / 3.0:
        return (0.15, 0.70, 1.00, 0.70)
    return (1.00, 1.00, 0.70, 0.25)


def right_half(x: torch.Tensor) -> torch.Tensor:
    return x[..., x.shape[-1] // 2 :]


def multiscale_right_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    progress: float,
    cfg: LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or LossConfig()
    pr = right_half(pred)
    tr = right_half(target)
    weights = curriculum_weights(progress, cfg.scales)

    total = pred.new_zeros(())
    norm = 0.0
    parts: dict[str, torch.Tensor] = {}
    for sigma, weight in zip(cfg.scales, weights):
        if weight <= 0:
            continue
        pb = gaussian_blur(pr, sigma)
        tb = gaussian_blur(tr, sigma)
        mse = (pb - tb).square().mean()
        energy = tb.square().mean().detach().clamp_min(cfg.eps)
        rel = mse / energy
        parts[f"scale_{sigma:g}"] = rel
        total = total + float(weight) * rel
        norm += float(weight)
    return total / max(norm, cfg.eps), parts


def organism_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    progress: float,
    cfg: LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or LossConfig()
    full = (pred - target).square().mean()
    pr = right_half(pred)
    tr = right_half(target)
    right = (pr - tr).square().mean()
    pyramid, parts = multiscale_right_loss(pred, target, progress=progress, cfg=cfg)
    mass = (pr.mean(dim=(-2, -1)) - tr.mean(dim=(-2, -1))).abs().mean()

    total = full + cfg.pyramid_gain * pyramid + cfg.mass_gain * mass
    metrics = {
        "full_mse": full,
        "right_mse": right,
        "pyramid": pyramid,
        "mass": mass,
        **parts,
    }
    return total, metrics
