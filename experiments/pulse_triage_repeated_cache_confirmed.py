"""Confirmed repeated-regression cache gate for PulseTriage.

The first execution of ``pulse_triage_repeated_cache`` exposed an assay defect:
the cached one-sided arm stopped after the sparse screen, while practical
PulseTriage always spends calls directly confirming a shortlist. Even the
fresh-reference ceiling therefore recovered only ~75% of lost KPI, below the
pre-registered 85% product boundary. That run did not cleanly test caching.

This repair changes exactly one architectural omission: keep the same 24 masks,
traffic drift, nuisance magnitude, seeds, TTL and locked thresholds, but restore
eight current-incident individual confirmations after the one-sided screen.
No healthy individual calibration is granted. The cache only supplies the 24
healthy group-mask references it was originally intended to supply.

Call accounting stays product-relevant:

    one-sided incident = 1 baseline + 24 groups + 8 confirms = 33 calls
    TTL-4 calibration = 25 healthy calls every four incidents = 6.25/incident
    total TTL-4       = 39.25 calls/incident = 68.86% of ordinary 57-call triage

The thresholds are unchanged from docs/PULSE_TRIAGE_CACHE.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pulsetriage import triage
from pulsetriage.core import _omp
from experiments.pulse_triage_repeated_cache import (
    FAULTS,
    INCIDENTS,
    LOCKED_GATE,
    SCREEN_PAIRS,
    SEQUENCES,
    SHORTLIST,
    TTL,
    EXPECTED_EFFECT_FRACTION,
    PublicIncidentEvaluator,
    _calibrate,
    _current_group_effects,
    _design,
    _epoch,
    _expected_effects,
    _new_bucket,
    _record,
    _summarize,
    _traffic_weights,
)
from experiments.pulse_triage_digits import DigitsRollbackWorld


def _screen_shortlist(
    matrix: np.ndarray,
    current_effects: np.ndarray,
    expected_effects: np.ndarray,
) -> list[int]:
    """Return the eight sparse-screen candidates before confirmation."""
    residual = current_effects - expected_effects
    effects, _ = _omp(matrix, residual, SHORTLIST)
    ranked = np.argsort(effects)[::-1]
    return [int(i) for i in ranked[:SHORTLIST]]


def _confirm(
    evaluator: PublicIncidentEvaluator,
    shortlist: list[int],
) -> tuple[set[str], int]:
    """Use the existing PulseTriage idea: current baseline + direct rollback.

    The group screen already bought the current baseline, so only the eight
    individual rollback calls are charged here. Healthy per-item nuisance is
    deliberately *not* supplied from the cache.
    """
    baseline = evaluator.score(frozenset(), count=False)
    measured: list[tuple[float, str]] = []
    for index in shortlist:
        item = evaluator.world.items[index]
        score = evaluator.score(frozenset((item,)))
        measured.append((score - baseline, item))
    measured.sort(reverse=True)
    selected = {item for gain, item in measured[:FAULTS] if gain > 0.0}
    return selected, len(shortlist)


def run(args: argparse.Namespace) -> dict:
    world = DigitsRollbackWorld(split_seed=args.split_seed, shards=4)
    matrix, masks = _design(len(world.items), args.design_seed)
    item_effects = _expected_effects(world, args.expected_seed)

    methods = {
        name: _new_bucket()
        for name in (
            "ordinary_pulsetriage",
            "uncalibrated_confirmed",
            "cache_once_confirmed",
            "ttl4_cache_confirmed",
            "fresh_reference_confirmed",
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

            if cache_once_effects is None:
                cache_once_effects, calls = _calibrate(
                    world, masks, item_effects, traffic
                )
                methods["cache_once_confirmed"]["healthy_calls"] += calls

            if ttl_effects is None or step % TTL == 0:
                ttl_effects, calls = _calibrate(world, masks, item_effects, traffic)
                methods["ttl4_cache_confirmed"]["healthy_calls"] += calls

            fresh_effects, calls = _calibrate(world, masks, item_effects, traffic)
            methods["fresh_reference_confirmed"]["healthy_calls"] += calls

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
                ("uncalibrated_confirmed", np.zeros(SCREEN_PAIRS, dtype=float)),
                ("cache_once_confirmed", cache_once_effects),
                ("ttl4_cache_confirmed", ttl_effects),
                ("fresh_reference_confirmed", fresh_effects),
            ):
                evaluator = evaluators[name]
                _baseline, current, screen_calls = _current_group_effects(evaluator, masks)
                shortlist = _screen_shortlist(
                    matrix,
                    current,
                    np.asarray(reference),
                )
                selected, confirm_calls = _confirm(evaluator, shortlist)
                methods[name]["incident_calls"] += screen_calls + confirm_calls
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

    ttl = measurements["ttl4_cache_confirmed"]
    ordinary = measurements["ordinary_pulsetriage"]
    uncalibrated = measurements["uncalibrated_confirmed"]
    cache_once = measurements["cache_once_confirmed"]
    fresh = measurements["fresh_reference_confirmed"]

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
        "assay": "confirmed repeated scalar-only regression diagnosis with cached healthy intervention consequences",
        "assay_repair": (
            "Restored the eight direct shortlist confirmations used by practical "
            "PulseTriage after the first screen-only run showed a 75% fresh-reference "
            "ceiling. Masks, drift, nuisance, seeds, TTL and locked thresholds unchanged."
        ),
        "dataset": "scikit-learn handwritten digits (8x8)",
        "model": "same frozen StandardScaler + multinomial logistic regression as PulseTriage receipt",
        "candidate_changes": len(world.items),
        "faults_per_incident": FAULTS,
        "screen_masks": SCREEN_PAIRS,
        "shortlist_confirmations": SHORTLIST,
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
    m = result["measurements"]
    assert result["candidate_changes"] == 256
    assert result["screen_masks"] == 24
    assert result["shortlist_confirmations"] == 8
    assert m["ordinary_pulsetriage"]["calls_per_incident"] == 57.0
    assert m["uncalibrated_confirmed"]["calls_per_incident"] == 33.0
    assert m["fresh_reference_confirmed"]["calls_per_incident"] == 58.0
    # Restoring the existing confirmation stage must actually repair the weak
    # one-sided ceiling before the cache/product classification is interpreted.
    assert m["fresh_reference_confirmed"]["score_recovery"]["mean"] >= 0.85
    assert m["ttl4_cache_confirmed"]["total_scalar_calls"] < m["ordinary_pulsetriage"]["total_scalar_calls"]


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
        print("CONFIRMED PULSE TRIAGE CACHE ASSAY EXECUTED")
    if args.require_pass and not all(result["gate_checks"].values()):
        raise SystemExit("confirmed cached PulseTriage did not clear locked product boundary")


if __name__ == "__main__":
    main()
