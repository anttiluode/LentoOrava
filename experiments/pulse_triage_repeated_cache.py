"""Repeated-regression cache gate for PulseTriage.

This keeps the executed handwritten-digits rollback workload but adds a small,
known-in-principle healthy rollback side-effect. The side-effect varies with
validation shard mix over time. A cached healthy intervention consequence can
therefore be subtracted from a later incident measurement.

The gate compares ordinary PulseTriage with a one-sided coded screen that uses
24 current group measurements plus remembered healthy references. All healthy
calibration calls are counted in the cumulative evidence budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pulsetriage import triage
from pulsetriage.core import _omp
from experiments.pulse_triage_digits import DigitsRollbackWorld


SCREEN_PAIRS = 24
SHORTLIST = 8
FAULTS = 4
INCIDENTS = 24
SEQUENCES = 8
TTL = 4
EXPECTED_EFFECT_FRACTION = 0.08

LOCKED_GATE = {
    "minimum_ttl_mean_score_recovery": 0.85,
    "minimum_ttl_mean_fault_recall": 0.75,
    "maximum_recovery_gap_vs_ordinary": 0.05,
    "maximum_ttl_fraction_of_ordinary_calls": 0.70,
}


@dataclass
class Incident:
    faults: np.ndarray
    traffic_weights: np.ndarray


def _traffic_weights(step: int) -> np.ndarray:
    """Stable -> traffic-mix drift -> new stable mix."""
    start = np.ones(4, dtype=float)
    end = np.asarray([1.45, 0.70, 1.20, 0.65], dtype=float)
    if step < 8:
        weights = start
    elif step < 16:
        alpha = (step - 7) / 8.0
        weights = (1.0 - alpha) * start + alpha * end
    else:
        weights = end
    return weights / np.mean(weights)


def _epoch(step: int) -> str:
    if step < 8:
        return "stable_pre"
    if step < 16:
        return "drift"
    return "stable_post"


def _design(n_items: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    matrix = np.zeros((SCREEN_PAIRS, n_items), dtype=float)
    masks: list[np.ndarray] = []
    indices = np.arange(n_items)
    for row in range(SCREEN_PAIRS):
        permutation = rng.permutation(indices)
        selected = np.sort(permutation[: n_items // 2])
        matrix[row, selected] = 1.0
        masks.append(selected)
    return matrix, masks


def _typical_fault_gain(world: DigitsRollbackWorld) -> float:
    gains = []
    clean = world.clean_score
    for index in world.fault_pool:
        logits = world.clean_logits - world.rollback_delta[int(index)]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_norm = np.log(np.sum(np.exp(shifted), axis=1))
        chosen = shifted[np.arange(len(world.labels)), world.labels]
        damaged = float(np.mean(chosen - log_norm))
        gains.append(max(clean - damaged, 0.0))
    return float(np.median(np.asarray(gains)))


def _expected_effects(world: DigitsRollbackWorld, seed: int) -> np.ndarray:
    """Small deterministic healthy rollback side-effects.

    The scale is locked relative to a typical true-fault repair in the actual
    classifier. Signs and item-to-item variation are fixed before incidents.
    """
    scale = max(_typical_fault_gain(world), 1e-6)
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, EXPECTED_EFFECT_FRACTION * scale, len(world.items))
    return effects.astype(float)


def _nuisance_effect(
    selected_indices: np.ndarray,
    item_effects: np.ndarray,
    traffic_weights: np.ndarray,
    shards: int,
) -> float:
    if len(selected_indices) == 0:
        return 0.0
    shard_ids = selected_indices % shards
    return float(np.sum(item_effects[selected_indices] * traffic_weights[shard_ids]))


class PublicIncidentEvaluator:
    """Real classifier fault response plus expected rollback collateral effect."""

    def __init__(
        self,
        world: DigitsRollbackWorld,
        fault_indices: np.ndarray,
        item_effects: np.ndarray,
        traffic_weights: np.ndarray,
    ):
        self.world = world
        self.base_trial = world.make_trial(fault_indices)
        self.item_effects = item_effects
        self.traffic_weights = traffic_weights
        self.calls = 0

    @property
    def fault_items(self) -> frozenset[str]:
        return self.base_trial.fault_items

    def score(self, rolled_back: frozenset[str], *, count: bool = True) -> float:
        if count:
            self.calls += 1
        base = self.base_trial.score(rolled_back, count=False)
        indices = np.asarray(
            [self.world.item_to_index[item] for item in rolled_back], dtype=int
        )
        nuisance = _nuisance_effect(
            indices,
            self.item_effects,
            self.traffic_weights,
            self.world.shards,
        )
        return float(base + nuisance)


class HealthyEvaluator:
    """Fault-free scalar evaluator used only during declared healthy intervals."""

    def __init__(
        self,
        world: DigitsRollbackWorld,
        item_effects: np.ndarray,
        traffic_weights: np.ndarray,
    ):
        self.world = world
        self.item_effects = item_effects
        self.traffic_weights = traffic_weights
        self.calls = 0

    def score_indices(self, indices: np.ndarray, *, count: bool = True) -> float:
        if count:
            self.calls += 1
        return float(
            self.world.clean_score
            + _nuisance_effect(
                indices,
                self.item_effects,
                self.traffic_weights,
                self.world.shards,
            )
        )

    def baseline(self, *, count: bool = True) -> float:
        if count:
            self.calls += 1
        return float(self.world.clean_score)


def _calibrate(
    world: DigitsRollbackWorld,
    masks: list[np.ndarray],
    item_effects: np.ndarray,
    traffic_weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    evaluator = HealthyEvaluator(world, item_effects, traffic_weights)
    baseline = evaluator.baseline()
    effects = np.asarray(
        [evaluator.score_indices(mask) - baseline for mask in masks], dtype=float
    )
    return effects, evaluator.calls


def _current_group_effects(
    evaluator: PublicIncidentEvaluator,
    masks: list[np.ndarray],
) -> tuple[float, np.ndarray, int]:
    baseline = evaluator.score(frozenset())
    values = []
    for mask in masks:
        rolled_back = frozenset(evaluator.world.items[int(i)] for i in mask)
        values.append(evaluator.score(rolled_back) - baseline)
    return baseline, np.asarray(values, dtype=float), 1 + len(masks)


def _screen_select(
    matrix: np.ndarray,
    current_effects: np.ndarray,
    expected_effects: np.ndarray,
    items: list[str],
    faults: int,
) -> set[str]:
    residual = current_effects - expected_effects
    effects, _ = _omp(matrix, residual, SHORTLIST)
    ranked = np.argsort(effects)[::-1]
    selected = [int(i) for i in ranked[:faults] if effects[int(i)] > 0.0]
    return {items[i] for i in selected}


def _recovery(evaluator: PublicIncidentEvaluator, selected: set[str]) -> float:
    baseline = evaluator.score(frozenset(), count=False)
    oracle = evaluator.score(evaluator.fault_items, count=False)
    chosen = evaluator.score(frozenset(selected), count=False)
    denominator = oracle - baseline
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip((chosen - baseline) / denominator, 0.0, 1.0))


def _summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "central_95pct": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def _record(
    bucket: dict,
    evaluator: PublicIncidentEvaluator,
    selected: set[str],
    epoch: str,
) -> None:
    recovery = _recovery(evaluator, selected)
    recall = len(selected & evaluator.fault_items) / FAULTS
    bucket["recovery"].append(recovery)
    bucket["recall"].append(recall)
    bucket["exact"].append(float(selected == set(evaluator.fault_items)))
    bucket["epoch_recovery"][epoch].append(recovery)
    bucket["epoch_recall"][epoch].append(recall)


def _new_bucket() -> dict:
    return {
        "recovery": [],
        "recall": [],
        "exact": [],
        "healthy_calls": 0,
        "incident_calls": 0,
        "epoch_recovery": {name: [] for name in ("stable_pre", "drift", "stable_post")},
        "epoch_recall": {name: [] for name in ("stable_pre", "drift", "stable_post")},
    }


def run(args: argparse.Namespace) -> dict:
    world = DigitsRollbackWorld(split_seed=args.split_seed, shards=4)
    matrix, masks = _design(len(world.items), args.design_seed)
    item_effects = _expected_effects(world, args.expected_seed)

    methods = {
        name: _new_bucket()
        for name in (
            "ordinary_pulsetriage",
            "uncalibrated_screen",
            "cache_once",
            "ttl4_cache",
            "fresh_reference",
        )
    }

    for sequence in range(args.sequences):
        rng = np.random.default_rng(args.seed + sequence * 10_007)
        cache_once_effects: np.ndarray | None = None
        ttl_effects: np.ndarray | None = None

        for step in range(args.incidents):
            epoch = _epoch(step)
            traffic = _traffic_weights(step)
            faults = rng.choice(world.fault_pool, size=FAULTS, replace=False)

            # Healthy calibration is acquired only when the policy says it is.
            if cache_once_effects is None:
                cache_once_effects, calls = _calibrate(
                    world, masks, item_effects, traffic
                )
                methods["cache_once"]["healthy_calls"] += calls

            if ttl_effects is None or step % TTL == 0:
                ttl_effects, calls = _calibrate(world, masks, item_effects, traffic)
                methods["ttl4_cache"]["healthy_calls"] += calls

            fresh_effects, calls = _calibrate(world, masks, item_effects, traffic)
            methods["fresh_reference"]["healthy_calls"] += calls

            # All methods see the same fault set and traffic mix, but maintain
            # separate call accounting/evaluator objects.
            evaluators = {
                name: PublicIncidentEvaluator(world, faults, item_effects, traffic)
                for name in methods
            }

            ordinary = evaluators["ordinary_pulsetriage"]
            result = triage(
                world.items,
                ordinary.score,
                budget=57,
                max_suspects=FAULTS,
                screening_pairs=SCREEN_PAIRS,
                shortlist_size=SHORTLIST,
                model_terms=SHORTLIST,
                permutations=0,
                seed=args.seed + sequence * 997 + step,
            )
            ordinary_selected = (
                {suspect.item for suspect in result.suspects}
                if result.status == "ok"
                else set()
            )
            methods["ordinary_pulsetriage"]["incident_calls"] += result.calls
            _record(methods["ordinary_pulsetriage"], ordinary, ordinary_selected, epoch)

            for name, reference in (
                ("uncalibrated_screen", np.zeros(SCREEN_PAIRS, dtype=float)),
                ("cache_once", cache_once_effects),
                ("ttl4_cache", ttl_effects),
                ("fresh_reference", fresh_effects),
            ):
                evaluator = evaluators[name]
                _baseline, current, calls = _current_group_effects(evaluator, masks)
                selected = _screen_select(
                    matrix,
                    current,
                    np.asarray(reference),
                    world.items,
                    FAULTS,
                )
                methods[name]["incident_calls"] += calls
                _record(methods[name], evaluator, selected, epoch)

    total_incidents = args.sequences * args.incidents
    measurements = {}
    for name, bucket in methods.items():
        total_calls = bucket["healthy_calls"] + bucket["incident_calls"]
        measurements[name] = {
            "score_recovery": _summarize(bucket["recovery"]),
            "fault_recall": _summarize(bucket["recall"]),
            "exact_fault_set_rate": float(np.mean(bucket["exact"])),
            "healthy_calibration_calls": int(bucket["healthy_calls"]),
            "incident_validation_calls": int(bucket["incident_calls"]),
            "total_scalar_calls": int(total_calls),
            "calls_per_incident": total_calls / total_incidents,
            "recovery_by_epoch": {
                epoch: float(np.mean(bucket["epoch_recovery"][epoch]))
                for epoch in bucket["epoch_recovery"]
            },
            "recall_by_epoch": {
                epoch: float(np.mean(bucket["epoch_recall"][epoch]))
                for epoch in bucket["epoch_recall"]
            },
        }

    ttl = measurements["ttl4_cache"]
    ordinary = measurements["ordinary_pulsetriage"]
    uncalibrated = measurements["uncalibrated_screen"]
    cache_once = measurements["cache_once"]
    fresh = measurements["fresh_reference"]

    gate_checks = {
        "ttl_score_recovery": bool(
            ttl["score_recovery"]["mean"]
            >= LOCKED_GATE["minimum_ttl_mean_score_recovery"]
        ),
        "ttl_fault_recall": bool(
            ttl["fault_recall"]["mean"]
            >= LOCKED_GATE["minimum_ttl_mean_fault_recall"]
        ),
        "ttl_near_ordinary_recovery": bool(
            ttl["score_recovery"]["mean"]
            >= ordinary["score_recovery"]["mean"]
            - LOCKED_GATE["maximum_recovery_gap_vs_ordinary"]
        ),
        "ttl_saves_calls": bool(
            ttl["total_scalar_calls"]
            <= LOCKED_GATE["maximum_ttl_fraction_of_ordinary_calls"]
            * ordinary["total_scalar_calls"]
        ),
        "cache_beats_uncalibrated": bool(
            ttl["score_recovery"]["mean"]
            >= uncalibrated["score_recovery"]["mean"] + 0.05
        ),
    }

    return {
        "assay": "repeated scalar-only regression diagnosis with cached healthy intervention consequences",
        "dataset": "scikit-learn handwritten digits (8x8)",
        "model": "same frozen StandardScaler + multinomial logistic regression as PulseTriage receipt",
        "candidate_changes": len(world.items),
        "faults_per_incident": FAULTS,
        "screen_masks": SCREEN_PAIRS,
        "sequences": args.sequences,
        "incidents_per_sequence": args.incidents,
        "total_incidents": total_incidents,
        "cache_ttl_incidents": TTL,
        "expected_effect_fraction_of_typical_fault_gain": EXPECTED_EFFECT_FRACTION,
        "traffic_schedule": "stable 0-7 -> shard-mix drift 8-15 -> new stable 16-23",
        "public_observation": "one aggregate validation KPI per rollback",
        "locked_gate": LOCKED_GATE,
        "measurements": measurements,
        "comparisons": {
            "ttl_fraction_of_ordinary_calls": ttl["total_scalar_calls"]
            / ordinary["total_scalar_calls"],
            "ttl_minus_uncalibrated_recovery": ttl["score_recovery"]["mean"]
            - uncalibrated["score_recovery"]["mean"],
            "ttl_minus_cache_once_recovery": ttl["score_recovery"]["mean"]
            - cache_once["score_recovery"]["mean"],
            "fresh_minus_ttl_recovery": fresh["score_recovery"]["mean"]
            - ttl["score_recovery"]["mean"],
        },
        "gate_checks": gate_checks,
        "classification": (
            "CACHED_HEALTHY_INTERVENTION_OUTCOMES_AMORTIZE_REPEATED_PULSE_TRIAGE"
            if all(gate_checks.values())
            else "CACHED_PULSE_TRIAGE_DOES_NOT_CLEAR_LOCKED_PRODUCT_BOUNDARY"
        ),
        "scope": (
            "The classifier/fault response is the executed real digits workload. "
            "Healthy rollback side-effects and traffic-mix drift are controlled "
            "nuisance terms added to test cross-incident calibration reuse."
        ),
    }


def check(result: dict) -> None:
    measurements = result["measurements"]
    assert result["candidate_changes"] == 256
    assert result["screen_masks"] == 24
    assert measurements["ordinary_pulsetriage"]["calls_per_incident"] == 57.0
    assert measurements["uncalibrated_screen"]["calls_per_incident"] == 25.0
    assert measurements["fresh_reference"]["calls_per_incident"] == 50.0
    # The test itself is allowed to reject the cache. These checks only lock
    # accounting and ensure the calibrated assay is informative enough to run.
    assert measurements["fresh_reference"]["score_recovery"]["mean"] >= 0.70
    assert measurements["ttl4_cache"]["total_scalar_calls"] < measurements["ordinary_pulsetriage"]["total_scalar_calls"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=SEQUENCES)
    parser.add_argument("--incidents", type=int, default=INCIDENTS)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--design-seed", type=int, default=9031)
    parser.add_argument("--expected-seed", type=int, default=7719)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.check:
        check(result)
        print("PULSE TRIAGE CACHE ASSAY EXECUTED")
    if args.require_pass and not all(result["gate_checks"].values()):
        raise SystemExit("cached PulseTriage did not clear locked product boundary")


if __name__ == "__main__":
    main()
