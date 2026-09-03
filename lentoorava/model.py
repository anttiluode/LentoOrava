from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OrganismConfig:
    image_size: int = 32
    channels: int = 24
    n_agents: int = 12
    hidden_dim: int = 48
    read_dim: int = 48
    patch_size: int = 3
    steps: int = 8
    max_step: float = 0.24
    write_sigma: float = 0.11
    write_gain: float = 0.45
    transport_gain: float = 0.25
    start_layout: str = "source_line"
    source_x: float = -0.548
    target_x: float = 0.548
    y_min: float = -0.677
    y_max: float = 0.677
    decode_bias: float = -4.0


class BoundedImageOrganism(nn.Module):
    """Shared latent field plus bounded local observer/writers."""

    def __init__(self, cfg: OrganismConfig | None = None, *, motion: str = "learned", agents: bool = True):
        super().__init__()
        self.cfg = cfg or OrganismConfig()
        self.motion = motion
        if motion not in {"learned", "fixed", "scripted"}:
            raise ValueError(f"unknown motion: {motion}")
        self.agents = bool(agents)
        c = self.cfg.channels
        p = self.cfg.patch_size

        self.in_proj = nn.Conv2d(3, c, kernel_size=1)
        self.transport = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )
        self.read_proj = nn.Sequential(
            nn.Linear(c * p * p, self.cfg.read_dim),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(self.cfg.read_dim + 2, self.cfg.hidden_dim)
        self.move_head = nn.Linear(self.cfg.hidden_dim, 2)
        self.write_head = nn.Linear(self.cfg.hidden_dim, c)
        self.gate_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.decode = nn.Conv2d(c, 3, kernel_size=1)
        nn.init.constant_(self.decode.bias, self.cfg.decode_bias)

        self.agent_seed = nn.Parameter(torch.zeros(self.cfg.n_agents, self.cfg.hidden_dim))
        nn.init.normal_(self.agent_seed, std=0.02)
        self.register_buffer("start_pos", self._initial_positions())
        self.register_buffer("pixel_grid", self._pixel_grid(self.cfg.image_size), persistent=False)

    def _initial_positions(self) -> torch.Tensor:
        n = self.cfg.n_agents
        if self.cfg.start_layout == "grid":
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            xs = torch.linspace(-0.8, 0.8, cols)
            ys = torch.linspace(-0.8, 0.8, rows)
            pts = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)[..., [1, 0]].reshape(-1, 2)
            return pts[:n]
        if self.cfg.start_layout == "source_line":
            ys = torch.linspace(self.cfg.y_min, self.cfg.y_max, n)
            return torch.stack([torch.full_like(ys, self.cfg.source_x), ys], dim=-1)
        if self.cfg.start_layout == "bridge":
            n_source = (n + 1) // 2
            n_target = n - n_source
            ys_source = torch.linspace(self.cfg.y_min, self.cfg.y_max, n_source)
            source = torch.stack([torch.full_like(ys_source, self.cfg.source_x), ys_source], dim=-1)
            if n_target == 0:
                return source
            ys_target = torch.linspace(self.cfg.y_min, self.cfg.y_max, n_target)
            target = torch.stack([torch.full_like(ys_target, self.cfg.target_x), ys_target], dim=-1)
            return torch.cat([source, target], dim=0)
        raise ValueError(f"unknown start_layout: {self.cfg.start_layout}")

    @staticmethod
    def _pixel_grid(size: int) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, size)
        xs = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1)

    def _sample_local(self, field: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, c, h, _ = field.shape
        n = pos.shape[1]
        p = self.cfg.patch_size
        radius = 2.0 / max(h - 1, 1)
        offs = torch.linspace(-radius, radius, p, device=field.device, dtype=field.dtype)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        local = torch.stack([ox, oy], dim=-1)
        grid = pos[:, :, None, None, :] + local[None, None]
        grid = grid.clamp(-1.0, 1.0).reshape(b, n * p, p, 2)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.reshape(b, c, n, p, p).permute(0, 2, 1, 3, 4)
        return sampled.reshape(b, n, c * p * p)

    def _local_writes(self, pos: torch.Tensor, values: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        grid = self.pixel_grid.to(device=values.device, dtype=values.dtype)
        delta = grid[None, None] - pos[:, :, None, None, :]
        d2 = (delta * delta).sum(dim=-1)
        mask = torch.exp(-0.5 * d2 / (self.cfg.write_sigma ** 2))
        mask = mask / (mask.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        splat = (
            values[:, :, :, None, None]
            * gates[:, :, :, None, None]
            * mask[:, :, None]
        ).sum(dim=1)
        return splat / math.sqrt(max(self.cfg.n_agents, 1))

    def forward(self, observed: torch.Tensor, *, return_trace: bool = False):
        b = observed.shape[0]
        field = self.in_proj(observed)
        pos = self.start_pos.to(observed).unsqueeze(0).expand(b, -1, -1).clone()
        state = self.agent_seed.to(observed).unsqueeze(0).expand(b, -1, -1).clone()
        traces = [pos.detach()]

        for step in range(self.cfg.steps):
            field = field + self.cfg.transport_gain * self.transport(field)
            if self.agents:
                patches = self._sample_local(field, pos)
                read = self.read_proj(patches)
                cell_in = torch.cat([read, pos], dim=-1).reshape(b * self.cfg.n_agents, -1)
                state = self.cell(
                    cell_in,
                    state.reshape(b * self.cfg.n_agents, -1),
                ).reshape(b, self.cfg.n_agents, -1)

                if self.motion == "learned":
                    move = torch.tanh(self.move_head(state)) * self.cfg.max_step
                    pos = (pos + move).clamp(-1.0, 1.0)
                elif self.motion == "scripted":
                    frac = float(step + 1) / float(self.cfg.steps)
                    x = self.cfg.source_x + frac * (self.cfg.target_x - self.cfg.source_x)
                    pos = torch.stack([torch.full_like(pos[..., 0], x), pos[..., 1]], dim=-1)

                values = torch.tanh(self.write_head(state))
                gates = torch.sigmoid(self.gate_head(state))
                field = field + self.cfg.write_gain * self._local_writes(pos, values, gates)

            traces.append(pos.detach())

        image = torch.sigmoid(self.decode(field))
        if return_trace:
            return image, torch.stack(traces, dim=1)
        return image


def make_variant(name: str, cfg: OrganismConfig | None = None) -> BoundedImageOrganism:
    name = name.lower()
    if name == "active":
        return BoundedImageOrganism(cfg, motion="learned", agents=True)
    if name == "carrier":
        return BoundedImageOrganism(cfg, motion="scripted", agents=True)
    if name == "fixed":
        return BoundedImageOrganism(cfg, motion="fixed", agents=True)
    if name == "transport":
        return BoundedImageOrganism(cfg, motion="fixed", agents=False)
    raise ValueError(f"unknown variant: {name}")
