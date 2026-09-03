"""Real-workload receipt for PulseTriage.

The workload is a frozen scikit-learn classifier on the handwritten-digits
dataset.  There are 256 possible deployed preprocessing changes: one for each
8x8 pixel feature and each of four validation-data shards.  Four changes are
silently broken and zero their pixel values.  A legal intervention rolls back
any requested subset of changes; the debugger receives only aggregate negative
log loss from the complete validation set.

The model, labels, per-example losses, and injected fault identities remain
inside the evaluator.  PulseTriage sees only candidate IDs and scalar scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pulsetriage import triage


LOCKED_GATE = {
    "minimum_mean_score_recovery": 0.85,
    "minimum_mean_fault_recall": 0.80,
    "minimum_recovery_advantage_over_random": 0.45,
    "maximum_fraction_of_exhaustive_calls": 0.25,
    "maximum_shuffled_address_recovery": 0.40,
}


def _log_probability_score(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    chosen = shifted[np.arange(len(labels)), labels]
    return float(np.mean(chosen - log_normalizer))


class DigitsRollbackWorld:
    """Frozen classifier plus reversible preprocessing-shard rollbacks."""

    def __init__(self, *, split_seed: int = 42, shards: int = 4):
        try:
            from sklearn.datasets import load_digits
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:  # pragma: no cover - exercised in user environments
            raise SystemExit(
                "This receipt needs scikit-learn: python -m pip install scikit-learn"
            ) from error

        features, labels = load_digits(return_X_y=True)
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=0.35,
            random_state=split_seed,
            stratify=labels,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs"),
        )
        model.fit(train_x, train_y)

        scaler = model[0]
        classifier = model[1]
        self.labels = test_y.astype(int)
        self.clean_logits = np.asarray(model.decision_function(test_x), dtype=float)
        clean_prediction = np.argmax(self.clean_logits, axis=1)
        self.clean_accuracy = float(accuracy_score(test_y, clean_prediction))
        self.clean_score = _log_probability_score(self.clean_logits, self.labels)
        self.shards = shards
        self.items = [
            f"pixel-{feature:02d}/shard-{shard}"
            for feature in range(test_x.shape[1])
            for shard in range(shards)
        ]
        self.item_to_index = {item: index for index, item in enumerate(self.items)}

        # For a linear classifier, rolling back a zeroed raw pixel adds this
        # exact contribution to the frozen logits.  The aggregate log-loss
        # score remains nonlinear, so rollback effects need not add perfectly.
        n_items = len(self.items)
        n_examples = len(test_x)
        n_classes = self.clean_logits.shape[1]
        self.rollback_delta = np.zeros((n_items, n_examples, n_classes), dtype=np.float32)
        row_ids = np.arange(n_examples)
        for feature in range(test_x.shape[1]):
            scale = float(scaler.scale_[feature])
            if scale <= 1e-12:
                continue
            per_row = (test_x[:, feature] / scale)[:, None] * classifier.coef_[:, feature]
            for shard in range(shards):
                item_index = feature * shards + shard
                selected = row_ids % shards == shard
                self.rollback_delta[item_index, selected] = per_row[selected]

        # Faults are sampled from informative features so the deployed KPI has
        # a real regression to diagnose.  The choice is random within this
        # declared pool on every trial; it is not optimized for PulseTriage.
        scaled_variation = np.std(train_x, axis=0) / np.maximum(scaler.scale_, 1e-12)
        importance = np.linalg.norm(classifier.coef_, axis=0) * scaled_variation
        variable = np.flatnonzero(np.std(train_x, axis=0) > 0.2)
        top_features = variable[np.argsort(importance[variable])[-24:]]
        self.fault_pool = np.asarray(
            [int(feature) * shards + shard for feature in top_features for shard in range(shards)],
            dtype=int,
        )

    def make_trial(self, fault_indices: np.ndarray) -> "DigitsTrial":
        return DigitsTrial(self, np.asarray(fault_indices, dtype=int))


class DigitsTrial:
    def __init__(self, world: DigitsRollbackWorld, fault_indices: np.ndarray):
        self.world = world
        self.fault_indices = tuple(int(index) for index in fault_indices)
        self.fault_items = frozenset(world.items[index] for index in self.fault_indices)
        total_damage = np.sum(world.rollback_delta[list(self.fault_indices)], axis=0)
        self.current_logits = world.clean_logits - total_damage
        self.calls = 0

    def score(self, rolled_back: frozenset[str], *, count: bool = True) -> float:
        if count:
            self.calls += 1
        active = [
            self.world.item_to_index[item]
            for item in rolled_back
            if item in self.fault_items
        ]
        logits = self.current_logits
        if active:
            logits = logits + np.sum(self.world.rollback_delta[active], axis=0)
        return _log_probability_score(logits, self.world.labels)


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    lo, hi = np.quantile(array, [0.025, 0.975])
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "central_95pct": [float(lo), float(hi)],
    }


def _recovery(trial: DigitsTrial, selected: set[str]) -> float:
    baseline = trial.score(frozenset(), count=False)
    ceiling = trial.world.clean_score
    selected_score = trial.score(frozenset(selected), count=False)
    denominator = max(ceiling - baseline, 1e-12)
    return float(np.clip((selected_score - baseline) / denominator, 0.0, 1.0))


def _individual_ruler(
    trial: DigitsTrial,
    candidates: list[str],
    *,
    probes: int,
    keep: int,
    rng: np.random.Generator | None,
) -> tuple[set[str], int]:
    baseline = trial.score(frozenset(), count=False)
    pool = np.asarray(candidates, dtype=object)
    if rng is not None:
        pool = rng.permutation(pool)
    pool = pool[:probes]
    measured = [
        (trial.score(frozenset((str(item),)), count=False) - baseline, str(item))
        for item in pool
    ]
    measured.sort(reverse=True)
    return {item for gain, item in measured[:keep] if gain > 0.0}, 1 + len(pool)


def run(args: argparse.Namespace) -> dict:
    world = DigitsRollbackWorld(split_seed=args.split_seed, shards=4)
    active_recovery: list[float] = []
    active_recall: list[float] = []
    active_exact: list[float] = []
    active_ok: list[float] = []
    active_p: list[float] = []
    random_recovery: list[float] = []
    random_recall: list[float] = []
    exhaustive_recovery: list[float] = []
    exhaustive_recall: list[float] = []
    shuffled_recovery: list[float] = []
    shuffled_recall: list[float] = []

    for trial_index in range(args.trials):
        rng = np.random.default_rng(args.seed + trial_index * 101)
        faults = rng.choice(world.fault_pool, size=args.faults, replace=False)
        trial = world.make_trial(faults)
        result = triage(
            world.items,
            trial.score,
            budget=args.budget,
            max_suspects=args.faults,
            screening_pairs=args.screening_pairs,
            shortlist_size=args.shortlist,
            model_terms=args.shortlist,
            seed=args.seed + trial_index * 17,
            permutations=args.permutations,
        )
        # Operational accounting honors the refusal signal: an inconclusive
        # run commits no rollback, even if the object contains a diagnostic
        # shortlist for inspection.
        chosen = (
            {suspect.item for suspect in result.suspects}
            if result.status == "ok"
            else set()
        )
        recall = len(chosen & trial.fault_items) / args.faults
        active_recovery.append(_recovery(trial, chosen))
        active_recall.append(recall)
        active_exact.append(float(chosen == set(trial.fault_items)))
        active_ok.append(float(result.status == "ok"))
        active_p.append(result.screen_p_value)

        random_selected, _ = _individual_ruler(
            trial,
            world.items,
            probes=args.budget - 1,
            keep=args.faults,
            rng=np.random.default_rng(args.seed + trial_index * 101 + 1),
        )
        random_recovery.append(_recovery(trial, random_selected))
        random_recall.append(len(random_selected & trial.fault_items) / args.faults)

        exhaustive_selected, _ = _individual_ruler(
            trial,
            world.items,
            probes=len(world.items),
            keep=args.faults,
            rng=None,
        )
        exhaustive_recovery.append(_recovery(trial, exhaustive_selected))
        exhaustive_recall.append(len(exhaustive_selected & trial.fault_items) / args.faults)

        # Address-shuffle attacker: preserve every requested group size but
        # return the pulse from a random group of that size.  Baseline remains
        # baseline.  This severs consequence from the intervention address.
        shuffled_trial = world.make_trial(faults)
        shuffled_rng = np.random.default_rng(args.seed + trial_index * 101 + 2)

        def shuffled_evaluator(requested: frozenset[str]) -> float:
            if not requested:
                return shuffled_trial.score(frozenset())
            substitute = shuffled_rng.choice(
                world.items, size=len(requested), replace=False
            ).tolist()
            return shuffled_trial.score(frozenset(substitute))

        shuffled_result = triage(
            world.items,
            shuffled_evaluator,
            budget=args.budget,
            max_suspects=args.faults,
            screening_pairs=args.screening_pairs,
            shortlist_size=args.shortlist,
            model_terms=args.shortlist,
            seed=args.seed + trial_index * 17,
            permutations=args.permutations,
        )
        shuffled_chosen = (
            {suspect.item for suspect in shuffled_result.suspects}
            if shuffled_result.status == "ok"
            else set()
        )
        shuffled_recovery.append(_recovery(trial, shuffled_chosen))
        shuffled_recall.append(
            len(shuffled_chosen & trial.fault_items) / args.faults
        )

    active_calls = 1 + 2 * args.screening_pairs + args.shortlist
    exhaustive_calls = 1 + len(world.items)
    measurements = {
        "pulse_triage": {
            "score_recovery": _summary(active_recovery),
            "fault_recall": _summary(active_recall),
            "exact_fault_set_rate": float(np.mean(active_exact)),
            "conclusive_rate": float(np.mean(active_ok)),
            "median_screen_p_value": float(np.median(active_p)),
            "scalar_calls": active_calls,
        },
        "equal_budget_random_individual": {
            "score_recovery": _summary(random_recovery),
            "fault_recall": _summary(random_recall),
            "scalar_calls": args.budget,
        },
        "exhaustive_individual": {
            "score_recovery": _summary(exhaustive_recovery),
            "fault_recall": _summary(exhaustive_recall),
            "scalar_calls": exhaustive_calls,
        },
        "shuffled_pulse_address": {
            "score_recovery": _summary(shuffled_recovery),
            "fault_recall": _summary(shuffled_recall),
            "scalar_calls": active_calls,
        },
        "oracle_true_fault_rollback": {
            "score_recovery": 1.0,
            "fault_recall": 1.0,
            "available_to_debugger": False,
        },
    }
    comparisons = {
        "active_minus_random_score_recovery": float(
            np.mean(active_recovery) - np.mean(random_recovery)
        ),
        "active_fraction_of_exhaustive_calls": active_calls / exhaustive_calls,
        "active_fraction_of_exhaustive_score_recovery": float(
            np.mean(active_recovery) / max(np.mean(exhaustive_recovery), 1e-12)
        ),
    }
    gate_checks = {
        "mean_score_recovery": bool(
            np.mean(active_recovery) >= LOCKED_GATE["minimum_mean_score_recovery"]
        ),
        "mean_fault_recall": bool(
            np.mean(active_recall) >= LOCKED_GATE["minimum_mean_fault_recall"]
        ),
        "advantage_over_random": bool(
            comparisons["active_minus_random_score_recovery"]
            >= LOCKED_GATE["minimum_recovery_advantage_over_random"]
        ),
        "call_fraction": bool(
            comparisons["active_fraction_of_exhaustive_calls"]
            <= LOCKED_GATE["maximum_fraction_of_exhaustive_calls"]
        ),
        "shuffled_address_fails": bool(
            np.mean(shuffled_recovery)
            <= LOCKED_GATE["maximum_shuffled_address_recovery"]
        ),
    }

    return {
        "assay": "scalar-only rollback localization in a frozen classifier pipeline",
        "dataset": "scikit-learn handwritten digits (8x8)",
        "model": "StandardScaler + multinomial logistic regression",
        "split_seed": args.split_seed,
        "clean_validation_accuracy": world.clean_accuracy,
        "clean_validation_negative_log_loss": world.clean_score,
        "trials": args.trials,
        "trial_seed": args.seed,
        "candidate_changes": len(world.items),
        "candidate_definition": "64 pixel preprocessing transforms x 4 validation shards",
        "faults_per_trial": args.faults,
        "fault_sampling": (
            "uniform without replacement from the top 24 coefficient-norm pixel "
            "features x four shards; the pool is fixed before trial outcomes"
        ),
        "public_observation": "one aggregate validation negative-log-loss score per rollback",
        "pulse_triage_configuration": {
            "budget": args.budget,
            "balanced_complement_pairs": args.screening_pairs,
            "shortlist_confirmations": args.shortlist,
            "sparse_model_terms": args.shortlist,
            "permutation_calibrations": args.permutations,
        },
        "locked_gate": LOCKED_GATE,
        "measurements": measurements,
        "comparisons": comparisons,
        "gate_checks": gate_checks,
        "classification": (
            "PULSE_TRIAGE_PASSES_REAL_CLASSIFIER_ROLLBACK_GATE"
            if all(gate_checks.values())
            else "PULSE_TRIAGE_FAILS_REAL_CLASSIFIER_ROLLBACK_GATE"
        ),
        "scope": (
            "Sparse rollback debugging with an approximately additive scalar KPI. "
            "Faults are injected into a real frozen classifier workload; this is not "
            "a claim to diagnose arbitrary interactions or production systems without "
            "a safe reversible intervention."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260903)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--faults", type=int, default=4)
    parser.add_argument("--budget", type=int, default=57)
    parser.add_argument("--screening-pairs", type=int, default=24)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    if args.require_pass and not all(result["gate_checks"].values()):
        raise SystemExit("locked PulseTriage gate failed")


if __name__ == "__main__":
    main()
