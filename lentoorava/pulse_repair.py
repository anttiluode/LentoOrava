"""Scalar-pulse fault localization for a spatial body.

This module is deliberately independent of the neural model.  It isolates the
claim made by the browser demo:

* a repairer may propose an addressed local intervention;
* the evaluator returns one scalar score and no spatial error map;
* adaptive region probes should find a compact fault more efficiently than
  equal-budget random local probes.

The concrete repair primitive is bilateral copying.  It is a known local rule,
not an oracle: a destination patch is replaced by the corresponding patch on
the intact side.  Only ``ScalarPulseOracle.measure`` can inspect the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Region:
    """Half-open pixel rectangle."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1 - 1) / 2, (self.y0 + self.y1 - 1) / 2)

    def is_leaf(self, tile_size: int) -> bool:
        return self.width <= tile_size and self.height <= tile_size

    def split(self) -> tuple["Region", "Region"]:
        """Bisect the longer side, keeping both children non-empty."""

        if self.width >= self.height and self.width > 1:
            xm = self.x0 + self.width // 2
            return (
                Region(self.x0, self.y0, xm, self.y1),
                Region(xm, self.y0, self.x1, self.y1),
            )
        if self.height > 1:
            ym = self.y0 + self.height // 2
            return (
                Region(self.x0, self.y0, self.x1, ym),
                Region(self.x0, ym, self.x1, self.y1),
            )
        return (self, self)


@dataclass(frozen=True)
class Probe:
    region: Region
    gain: float
    depth: int


@dataclass
class SearchResult:
    selected: list[Probe]
    probes: list[Probe]
    pulse_calls: int
    initial_score: float


class ScalarPulseOracle:
    """Black-box evaluator whose public observation is a single float."""

    def __init__(self, target: Array):
        target = np.asarray(target, dtype=np.float32)
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError("target must have shape [height, width, 3]")
        self.__target = target.copy()
        self.calls = 0

    def measure(self, state: Array) -> float:
        """Return one global pulse: negative total squared error."""

        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.__target.shape:
            raise ValueError("state and target must have equal shapes")
        self.calls += 1
        return -float(np.square(state - self.__target).sum(dtype=np.float64))

    def display_health(self, state: Array) -> float:
        """Human-facing 0..100 health meter; not used by search."""

        err = float(np.square(np.asarray(state) - self.__target).sum(dtype=np.float64))
        scale = float(np.square(self.__target).sum(dtype=np.float64)) + 1e-12
        return 100.0 * max(0.0, 1.0 - err / scale)


def make_symmetric_body(size: int = 96) -> Array:
    """Create the procedural bilateral body used by the receipt."""

    if size < 32:
        raise ValueError("size must be at least 32")
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    ax = np.abs(xx)

    background = np.zeros((size, size, 3), dtype=np.float32)
    background[...] = (0.012, 0.027, 0.040)

    body = np.exp(-((xx / 0.14) ** 2 + (yy / 0.70) ** 2) * 2.2)
    head = np.exp(-((xx / 0.22) ** 2 + ((yy + 0.61) / 0.20) ** 2) * 2.0)
    wing_shape = ((ax / 0.86) ** 1.65 + ((yy + 0.02) / 0.72) ** 2) < 1.0
    shoulder_cut = ax > (0.12 + 0.18 * (yy + 0.7).clip(0, 1.4))
    wing = wing_shape & shoulder_cut
    wing_fade = np.clip(1.0 - (ax / 0.92) ** 2 - ((yy + 0.02) / 0.77) ** 2, 0, 1)

    veins = (
        np.exp(-((np.sin(18 * ax + 4.5 * yy) * 0.55) ** 2) / 0.025)
        * np.exp(-((ax - 0.52) / 0.38) ** 2)
        * wing
    )
    ribs = (
        np.exp(-((np.sin(14 * yy - 5 * ax) * 0.55) ** 2) / 0.035)
        * np.exp(-((ax - 0.48) / 0.42) ** 2)
        * wing
    )
    membrane = wing_fade * wing

    target = background.copy()
    target[..., 0] += 0.06 * membrane + 0.20 * veins + 0.05 * ribs
    target[..., 1] += 0.20 * membrane + 0.80 * veins + 0.34 * ribs
    target[..., 2] += 0.27 * membrane + 0.88 * veins + 0.80 * ribs
    target[..., 0] += 0.20 * body + 0.25 * head
    target[..., 1] += 0.64 * body + 0.58 * head
    target[..., 2] += 0.70 * body + 0.46 * head

    eye_y = -0.64
    for eye_x in (-0.085, 0.085):
        eye = np.exp(-(((xx - eye_x) / 0.028) ** 2 + ((yy - eye_y) / 0.032) ** 2) * 2.0)
        target[..., 0] += 0.92 * eye
        target[..., 1] += 0.84 * eye
        target[..., 2] += 0.32 * eye

    return np.clip(target, 0.0, 1.0).astype(np.float32)


def apply_wounds(
    state: Array,
    wounds: Iterable[tuple[float, float, float]],
    *,
    floor: tuple[float, float, float] = (0.012, 0.027, 0.040),
) -> Array:
    """Erase soft circular wounds specified in normalized coordinates."""

    result = np.asarray(state, dtype=np.float32).copy()
    h, w, _ = result.shape
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    floor_arr = np.asarray(floor, dtype=np.float32)
    for x, y, radius in wounds:
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = np.clip((radius - distance) / max(radius * 0.28, 1e-6), 0.0, 1.0)
        result = result * (1.0 - mask[..., None]) + floor_arr * mask[..., None]
    return result.astype(np.float32)


def mirrored_trial(state: Array, region: Region, *, alpha: float = 1.0) -> Array:
    """Tentatively copy the corresponding opposite-side pixels into region."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    result = np.asarray(state, dtype=np.float32).copy()
    width = result.shape[1]
    source_indices = width - 1 - np.arange(region.x0, region.x1)
    source = result[region.y0 : region.y1, source_indices, :].copy()
    destination = result[region.y0 : region.y1, region.x0 : region.x1, :]
    result[region.y0 : region.y1, region.x0 : region.x1, :] = (
        (1.0 - alpha) * destination + alpha * source
    )
    return result


def probe_region(
    oracle: ScalarPulseOracle,
    state: Array,
    baseline: float,
    region: Region,
    *,
    alpha: float = 0.35,
) -> float:
    """Return scalar consequence of one reversible addressed intervention."""

    return oracle.measure(mirrored_trial(state, region, alpha=alpha)) - baseline


def adaptive_pulse_search(
    state: Array,
    oracle: ScalarPulseOracle,
    *,
    budget: int = 36,
    repair_slots: int = 4,
    tile_size: int = 6,
    alpha: float = 0.35,
    root: Region | None = None,
) -> SearchResult:
    """Best-first hierarchical group testing using only scalar probe gains."""

    h, w, _ = state.shape
    root = root or Region(w // 2, 0, w, h)
    start_calls = oracle.calls
    baseline = oracle.measure(state)
    probes: list[Probe] = []
    leaves: list[Probe] = []
    queue: list[tuple[float, int, int, Probe]] = []
    serial = 0

    root_gain = probe_region(oracle, state, baseline, root, alpha=alpha)
    root_probe = Probe(root, root_gain, 0)
    probes.append(root_probe)
    heapq.heappush(queue, (-max(root_gain, 0.0), root.area, serial, root_probe))
    serial += 1

    while queue and oracle.calls - start_calls < budget:
        _, _, _, parent = heapq.heappop(queue)
        if parent.gain <= 0:
            continue
        if parent.region.is_leaf(tile_size):
            leaves.append(parent)
            continue

        children = parent.region.split()
        if children[0] == children[1]:
            leaves.append(parent)
            continue
        for child in children:
            if oracle.calls - start_calls >= budget:
                break
            gain = probe_region(oracle, state, baseline, child, alpha=alpha)
            probe = Probe(child, gain, parent.depth + 1)
            probes.append(probe)
            if child.is_leaf(tile_size):
                if gain > 0:
                    leaves.append(probe)
            elif gain > 0:
                heapq.heappush(queue, (-gain, child.area, serial, probe))
                serial += 1

    # If a compact wound straddles tile boundaries, nearby leaves may overlap in
    # value but not space.  We simply rank the causally measured leaf actions.
    leaves.sort(key=lambda item: item.gain, reverse=True)
    return SearchResult(
        selected=leaves[:repair_slots],
        probes=probes,
        pulse_calls=oracle.calls - start_calls,
        initial_score=baseline,
    )


def leaf_grid(root: Region, tile_size: int) -> list[Region]:
    """Enumerate non-overlapping local actions for the random attacker."""

    leaves: list[Region] = []
    for y0 in range(root.y0, root.y1, tile_size):
        for x0 in range(root.x0, root.x1, tile_size):
            leaves.append(
                Region(
                    x0,
                    y0,
                    min(x0 + tile_size, root.x1),
                    min(y0 + tile_size, root.y1),
                )
            )
    return leaves


def random_pulse_search(
    state: Array,
    oracle: ScalarPulseOracle,
    *,
    budget: int = 36,
    repair_slots: int = 4,
    tile_size: int = 6,
    alpha: float = 0.35,
    root: Region | None = None,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    """Equal-pulse attacker: sample local tiles uniformly, retain best gains."""

    h, w, _ = state.shape
    root = root or Region(w // 2, 0, w, h)
    rng = rng or np.random.default_rng(0)
    start_calls = oracle.calls
    baseline = oracle.measure(state)
    candidates = leaf_grid(root, tile_size)
    rng.shuffle(candidates)
    probes: list[Probe] = []

    # One call is consumed by the cached baseline in both methods.
    for region in candidates[: max(0, budget - 1)]:
        gain = probe_region(oracle, state, baseline, region, alpha=alpha)
        probes.append(Probe(region, gain, 0))
    ranked = sorted((p for p in probes if p.gain > 0), key=lambda p: p.gain, reverse=True)
    return SearchResult(
        selected=ranked[:repair_slots],
        probes=probes,
        pulse_calls=oracle.calls - start_calls,
        initial_score=baseline,
    )


def apply_selected_repairs(state: Array, selected: Iterable[Probe]) -> Array:
    """Commit the local writes chosen from reversible pulse probes."""

    result = np.asarray(state, dtype=np.float32).copy()
    for probe in selected:
        result = mirrored_trial(result, probe.region, alpha=1.0)
    return result


def recovered_fraction(oracle: ScalarPulseOracle, damaged: Array, repaired: Array) -> float:
    """Fraction of the initial scalar-score deficit recovered."""

    damaged_score = oracle.measure(damaged)
    repaired_score = oracle.measure(repaired)
    denominator = max(-damaged_score, 1e-12)
    return float(np.clip((repaired_score - damaged_score) / denominator, 0.0, 1.0))

