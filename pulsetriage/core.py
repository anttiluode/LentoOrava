"""Find sparse harmful changes with reversible rollbacks and one scalar score.

``triage`` never sees gradients, logs, internals, or a spatial error map.  Its
only experiment is: roll back a chosen subset of candidate changes and ask the
caller for one number.  Balanced complementary rollback masks form a compact
screening design; orthogonal matching pursuit decodes a sparse shortlist; and
individual rollbacks confirm the final suspects.

The useful assumption is deliberately narrow: only a few candidates are
harmful and grouped rollback effects are approximately additive.  The result
includes a permutation-calibrated screen test so a caller can refuse a ranking
when the scalar responses do not contain stable sparse signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np


ScoreFunction = Callable[[frozenset[str]], float]


@dataclass(frozen=True)
class ProbeRecord:
    """One call to the caller-owned scalar evaluator."""

    kind: str
    rolled_back: tuple[str, ...]
    raw_score: float
    quality_score: float


@dataclass(frozen=True)
class Suspect:
    """A candidate that survived coded screening and direct confirmation."""

    item: str
    screen_effect: float
    confirmed_effect: float


@dataclass
class TriageResult:
    """Auditable output of a PulseTriage run."""

    status: str
    candidates: int
    budget: int
    calls: int
    higher_is_better: bool
    baseline_score: float
    screening_pairs: int
    model_terms: int
    shortlist_size: int
    screen_fit_r2: float
    screen_null_fit_r2_95: float
    screen_p_value: float
    min_confirmed_effect: float
    suspects: list[Suspect]
    shortlist: list[Suspect]
    warnings: list[str] = field(default_factory=list)
    probes: list[ProbeRecord] = field(default_factory=list)

    def to_dict(self, *, include_probes: bool = True) -> dict:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        if not include_probes:
            payload.pop("probes", None)
        return payload


def _quality(raw_score: float, higher_is_better: bool) -> float:
    return raw_score if higher_is_better else -raw_score


def _omp(matrix: np.ndarray, outcomes: np.ndarray, terms: int) -> tuple[np.ndarray, float]:
    """Small dependency-free orthogonal matching pursuit decoder."""

    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.zeros(columns, dtype=float), 0.0

    chosen: list[int] = []
    residual = outcomes.astype(float, copy=True)
    coefficients = np.empty(0, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)

    for _ in range(min(terms, rows - 1, columns)):
        correlations = np.abs(matrix.T @ residual) / safe_norms
        if chosen:
            correlations[np.asarray(chosen)] = -np.inf
        column = int(np.argmax(correlations))
        if not np.isfinite(correlations[column]) or correlations[column] < 1e-12:
            break
        chosen.append(column)
        active = matrix[:, chosen]
        coefficients = np.linalg.lstsq(active, outcomes, rcond=None)[0]
        residual = outcomes - active @ coefficients

    effects = np.zeros(columns, dtype=float)
    if chosen:
        effects[np.asarray(chosen)] = coefficients

    prediction = matrix @ effects
    denominator = float(np.sum((outcomes - np.mean(outcomes)) ** 2))
    if denominator <= 1e-15:
        fit_r2 = 0.0
    else:
        fit_r2 = 1.0 - float(np.sum((outcomes - prediction) ** 2)) / denominator
    return effects, fit_r2


def _screen_significance(
    matrix: np.ndarray,
    outcomes: np.ndarray,
    terms: int,
    observed_r2: float,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compare sparse fit against fits obtained after severing pulse addresses."""

    if permutations <= 0:
        return float("nan"), float("nan")
    null_r2 = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(outcomes)
        _, null_r2[index] = _omp(matrix, shuffled, terms)
    p_value = (1.0 + float(np.sum(null_r2 >= observed_r2))) / (permutations + 1.0)
    return float(np.quantile(null_r2, 0.95)), p_value


def triage(
    items: Sequence[str],
    evaluate: ScoreFunction,
    *,
    budget: int = 57,
    max_suspects: int = 4,
    screening_pairs: int | None = None,
    shortlist_size: int | None = None,
    model_terms: int | None = None,
    higher_is_better: bool = True,
    baseline_repeats: int = 1,
    screen_repeats: int = 1,
    confirm_repeats: int = 1,
    min_effect: float | None = None,
    permutations: int = 128,
    seed: int = 0,
) -> TriageResult:
    """Localize sparse regressions under a strict scalar-evaluation budget.

    ``evaluate`` receives the set of candidate identifiers to roll back and
    returns one finite scalar score.  By default, larger values are better.  Set
    ``higher_is_better=False`` for metrics such as latency or error.

    The evaluator must make independent, reversible trials against the same
    deployed baseline.  Arbitrary high-order interactions are outside the
    method's contract; a weak permutation-calibrated screen and failed direct
    confirmations are reported as an inconclusive result rather than hidden.
    """

    ordered_items = tuple(str(item) for item in items)
    if len(ordered_items) < 2:
        raise ValueError("at least two candidate items are required")
    if len(set(ordered_items)) != len(ordered_items):
        raise ValueError("candidate item identifiers must be unique")
    if budget < 7:
        raise ValueError("budget must be at least 7 scalar evaluations")
    if max_suspects < 1:
        raise ValueError("max_suspects must be positive")
    for name, value in {
        "baseline_repeats": baseline_repeats,
        "screen_repeats": screen_repeats,
        "confirm_repeats": confirm_repeats,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")

    n_items = len(ordered_items)
    shortlist_size = shortlist_size or min(n_items, max(2 * max_suspects, max_suspects + 2))
    shortlist_size = min(n_items, max(max_suspects, shortlist_size))
    model_terms = model_terms or shortlist_size
    model_terms = min(n_items, max(max_suspects, model_terms))

    reserved = baseline_repeats + shortlist_size * confirm_repeats
    pair_cost = 2 * screen_repeats
    available_pairs = (budget - reserved) // pair_cost
    if screening_pairs is None:
        screening_pairs = available_pairs
    if screening_pairs < 2:
        raise ValueError("budget leaves room for fewer than two screening pairs")
    required = reserved + screening_pairs * pair_cost
    if required > budget:
        raise ValueError(
            f"configuration needs {required} evaluations but budget is {budget}"
        )

    probes: list[ProbeRecord] = []
    calls = 0

    def call(rolled_back: Sequence[str], kind: str) -> tuple[float, float]:
        nonlocal calls
        subset = frozenset(rolled_back)
        raw = float(evaluate(subset))
        if not np.isfinite(raw):
            raise ValueError(f"evaluator returned a non-finite score for {kind}")
        quality = _quality(raw, higher_is_better)
        probes.append(
            ProbeRecord(
                kind=kind,
                rolled_back=tuple(sorted(subset)),
                raw_score=raw,
                quality_score=quality,
            )
        )
        calls += 1
        return raw, quality

    baseline_samples = [call((), "baseline")[1] for _ in range(baseline_repeats)]
    baseline_quality = float(np.mean(baseline_samples))
    baseline_raw = baseline_quality if higher_is_better else -baseline_quality

    rng = np.random.default_rng(seed)
    matrix = np.zeros((screening_pairs, n_items), dtype=float)
    outcomes = np.zeros(screening_pairs, dtype=float)
    indices = np.arange(n_items)

    for row in range(screening_pairs):
        permutation = rng.permutation(indices)
        half = n_items // 2
        plus_indices = permutation[:half]
        minus_indices = permutation[half : 2 * half]
        matrix[row, plus_indices] = 1.0
        matrix[row, minus_indices] = -1.0
        plus = tuple(ordered_items[index] for index in plus_indices)
        minus = tuple(ordered_items[index] for index in minus_indices)

        plus_scores = [call(plus, "screen_plus")[1] for _ in range(screen_repeats)]
        minus_scores = [call(minus, "screen_minus")[1] for _ in range(screen_repeats)]
        outcomes[row] = float(np.mean(plus_scores) - np.mean(minus_scores))

    effects, fit_r2 = _omp(matrix, outcomes, model_terms)
    null_r2_95, p_value = _screen_significance(
        matrix,
        outcomes,
        model_terms,
        fit_r2,
        permutations=permutations,
        rng=np.random.default_rng(seed + 1_000_003),
    )

    ranked = np.argsort(effects)[::-1]
    shortlist_indices = ranked[:shortlist_size]
    baseline_std = (
        float(np.std(baseline_samples, ddof=1)) if len(baseline_samples) > 1 else 0.0
    )
    if min_effect is None:
        min_effect = 3.0 * baseline_std * np.sqrt(
            1.0 / baseline_repeats + 1.0 / confirm_repeats
        )

    shortlist: list[Suspect] = []
    for index in shortlist_indices:
        item = ordered_items[int(index)]
        confirmation = [call((item,), "confirm")[1] for _ in range(confirm_repeats)]
        confirmed_effect = float(np.mean(confirmation) - baseline_quality)
        shortlist.append(
            Suspect(
                item=item,
                screen_effect=float(effects[int(index)]),
                confirmed_effect=confirmed_effect,
            )
        )

    shortlist.sort(key=lambda candidate: candidate.confirmed_effect, reverse=True)
    suspects = [
        candidate for candidate in shortlist if candidate.confirmed_effect > float(min_effect)
    ][:max_suspects]

    warnings: list[str] = []
    status = "ok"
    if np.isfinite(p_value) and p_value > 0.10:
        status = "inconclusive"
        warnings.append(
            "Coded pulse addresses were not significant against shuffled outcomes; "
            "do not trust the ranking without more evaluations."
        )
    if not suspects:
        status = "inconclusive"
        warnings.append(
            "No shortlisted candidate improved the baseline beyond the confirmation threshold."
        )
    elif len(suspects) < max_suspects:
        warnings.append(
            f"Only {len(suspects)} candidate(s) cleared the confirmation threshold."
        )
    if screening_pairs < 2 * max_suspects:
        warnings.append(
            "The coded screen is very small relative to the requested sparse support."
        )
    if budget >= n_items + baseline_repeats:
        warnings.append(
            "The budget can afford exhaustive individual rollback; prefer that simpler ruler."
        )

    return TriageResult(
        status=status,
        candidates=n_items,
        budget=budget,
        calls=calls,
        higher_is_better=higher_is_better,
        baseline_score=baseline_raw,
        screening_pairs=screening_pairs,
        model_terms=model_terms,
        shortlist_size=shortlist_size,
        screen_fit_r2=float(fit_r2),
        screen_null_fit_r2_95=float(null_r2_95),
        screen_p_value=float(p_value),
        min_confirmed_effect=float(min_effect),
        suspects=suspects,
        shortlist=shortlist,
        warnings=warnings,
        probes=probes,
    )
