from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairCopyConfig:
    size: int = 32
    sigma: float = 1.8
    noise_std: float = 0.02
    left_x: int = 7
    right_x: int = 24
    y_margin: int = 5


class PairCopyDataset(Dataset):
    """Synthetic long-range image completion task."""

    _PALETTE = torch.tensor(
        [
            [1.0, 0.15, 0.15],
            [0.15, 1.0, 0.15],
            [0.15, 0.25, 1.0],
            [1.0, 0.85, 0.15],
            [0.9, 0.2, 0.9],
            [0.15, 0.9, 0.9],
        ],
        dtype=torch.float32,
    )

    def __init__(self, n: int = 4096, seed: int = 0, cfg: PairCopyConfig | None = None):
        self.n = int(n)
        self.seed = int(seed)
        self.cfg = cfg or PairCopyConfig()
        y = torch.arange(self.cfg.size, dtype=torch.float32)
        x = torch.arange(self.cfg.size, dtype=torch.float32)
        self.yy, self.xx = torch.meshgrid(y, x, indexing="ij")

    @property
    def y_values(self) -> range:
        return range(self.cfg.y_margin, self.cfg.size - self.cfg.y_margin)

    @property
    def n_colors(self) -> int:
        return len(self._PALETTE)

    def __len__(self) -> int:
        return self.n

    def _blob(self, x0: float, y0: float) -> torch.Tensor:
        s2 = 2.0 * self.cfg.sigma * self.cfg.sigma
        return torch.exp(-((self.xx - x0) ** 2 + (self.yy - y0) ** 2) / s2)

    def render(self, y0: int, color_idx: int, *, noisy: bool = False, generator=None):
        if y0 not in self.y_values:
            raise ValueError(f"y0={y0} outside allowed range {self.y_values}")
        if not (0 <= color_idx < self.n_colors):
            raise ValueError(f"color_idx={color_idx} outside [0,{self.n_colors})")
        color = self._PALETTE[color_idx]
        left = self._blob(self.cfg.left_x, y0)
        right = self._blob(self.cfg.right_x, y0)
        target = color[:, None, None] * (left + right).clamp(max=1.0)[None]
        observed = color[:, None, None] * left[None]
        if noisy and self.cfg.noise_std > 0:
            if generator is None:
                generator = torch.Generator().manual_seed(0)
            noise = torch.randn(observed.shape, generator=generator) * self.cfg.noise_std
            observed = (observed + noise).clamp(0.0, 1.0)
        return observed, target

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + int(idx) * 7919)
        y0 = int(torch.randint(self.cfg.y_margin, self.cfg.size - self.cfg.y_margin, (1,), generator=g))
        color_idx = int(torch.randint(0, len(self._PALETTE), (1,), generator=g))
        observed, target = self.render(y0, color_idx, noisy=True, generator=g)
        meta = {
            "y": torch.tensor(y0, dtype=torch.long),
            "color": torch.tensor(color_idx, dtype=torch.long),
        }
        return observed, target, meta
