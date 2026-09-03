"""Benchmark adaptive scalar-pulse localization against boring attackers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lentoorava.pulse_repair import (
    Region,
    ScalarPulseOracle,
    adaptive_pulse_search,
    apply_selected_repairs,
    apply_wounds,
    leaf_grid,
    make_symmetric_body,
    probe_region,
    random_pulse_search,
    recovered_fraction,
    Probe,
    SearchResult,
)


def sample_wounds(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    return [
        (
            float(rng.uniform(0.25, 0.82)),
            float(rng.uniform(-0.58, 0.58)),
            float(rng.uniform(0.09, 0.18)),
        )
        for _ in range(int(rng.integers(1, 4)))
    ]


def exhaustive_search(state, oracle, *, repair_slots: int, tile_size: int, alpha: float):
    h, w, _ = state.shape
    root = Region(w // 2, 0, w, h)
    start_calls = oracle.calls
    baseline = oracle.measure(state)
    probes = [
        Probe(region, probe_region(oracle, state, baseline, region, alpha=alpha), 0)
        for region in leaf_grid(root, tile_size)
    ]
    ranked = sorted((p for p in probes if p.gain > 0), key=lambda p: p.gain, reverse=True)
    return SearchResult(
        selected=ranked[:repair_slots],
        probes=probes,
        pulse_calls=oracle.calls - start_calls,
        initial_score=baseline,
    )


def interval(values: list[float]) -> list[float]:
    lo, hi = np.quantile(np.asarray(values), [0.025, 0.975])
    return [float(lo), float(hi)]


def summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "central_95pct": interval(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument("--seed", type=int, default=240903)
    ap.add_argument("--budget", type=int, default=36)
    ap.add_argument("--repair-slots", type=int, default=5)
    ap.add_argument("--tile-size", type=int, default=6)
    ap.add_argument("--probe-alpha", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    target = make_symmetric_body(96)
    active_values: list[float] = []
    random_values: list[float] = []
    exhaustive_values: list[float] = []
    active_calls: list[int] = []
    random_calls: list[int] = []
    exhaustive_calls: list[int] = []

    for trial in range(args.trials):
        wound_rng = np.random.default_rng(args.seed + trial * 17)
        damaged = apply_wounds(target, sample_wounds(wound_rng))

        active_oracle = ScalarPulseOracle(target)
        active = adaptive_pulse_search(
            damaged,
            active_oracle,
            budget=args.budget,
            repair_slots=args.repair_slots,
            tile_size=args.tile_size,
            alpha=args.probe_alpha,
        )
        active_repaired = apply_selected_repairs(damaged, active.selected)
        active_values.append(recovered_fraction(active_oracle, damaged, active_repaired))
        active_calls.append(active.pulse_calls)

        random_oracle = ScalarPulseOracle(target)
        random = random_pulse_search(
            damaged,
            random_oracle,
            budget=args.budget,
            repair_slots=args.repair_slots,
            tile_size=args.tile_size,
            alpha=args.probe_alpha,
            rng=np.random.default_rng(args.seed + trial * 17 + 1),
        )
        random_repaired = apply_selected_repairs(damaged, random.selected)
        random_values.append(recovered_fraction(random_oracle, damaged, random_repaired))
        random_calls.append(random.pulse_calls)

        exhaustive_oracle = ScalarPulseOracle(target)
        exhaustive = exhaustive_search(
            damaged,
            exhaustive_oracle,
            repair_slots=args.repair_slots,
            tile_size=args.tile_size,
            alpha=args.probe_alpha,
        )
        exhaustive_repaired = apply_selected_repairs(damaged, exhaustive.selected)
        exhaustive_values.append(
            recovered_fraction(exhaustive_oracle, damaged, exhaustive_repaired)
        )
        exhaustive_calls.append(exhaustive.pulse_calls)

    active_array = np.asarray(active_values)
    random_array = np.asarray(random_values)
    exhaustive_array = np.asarray(exhaustive_values)
    result = {
        "assay": "bilateral local repair under scalar-only global evaluation",
        "trials": args.trials,
        "seed": args.seed,
        "wounds_per_trial": [1, 3],
        "pulse_budget": args.budget,
        "repair_slots": args.repair_slots,
        "tile_size_px": args.tile_size,
        "probe_alpha": args.probe_alpha,
        "candidate_tiles": len(leaf_grid(Region(48, 0, 96, 96), args.tile_size)),
        "active": {
            **summary(active_values),
            "mean_pulse_calls": float(np.mean(active_calls)),
        },
        "equal_budget_random": {
            **summary(random_values),
            "mean_pulse_calls": float(np.mean(random_calls)),
        },
        "exhaustive_tile_scan": {
            **summary(exhaustive_values),
            "mean_pulse_calls": float(np.mean(exhaustive_calls)),
        },
        "paired": {
            "active_win_rate_vs_random": float(np.mean(active_array > random_array)),
            "active_tie_rate_vs_random": float(np.mean(active_array == random_array)),
            "mean_active_minus_random": float(np.mean(active_array - random_array)),
            "active_fraction_of_exhaustive_recovery": float(
                np.mean(active_array) / max(np.mean(exhaustive_array), 1e-12)
            ),
            "active_fraction_of_exhaustive_calls": float(
                np.mean(active_calls) / max(np.mean(exhaustive_calls), 1e-12)
            ),
        },
        "scope": (
            "Controlled sparse-fault assay with a valid bilateral repair primitive; "
            "not a claim of general image restoration or biological learning."
        ),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(out)


if __name__ == "__main__":
    main()

