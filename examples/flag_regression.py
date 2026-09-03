"""Thirty-second PulseTriage example with 256 candidate feature flags."""

from __future__ import annotations

from pulsetriage import triage


ITEMS = [f"flag-{index:03d}" for index in range(256)]
HARMFUL = {
    "flag-017": 1.7,
    "flag-083": 0.9,
    "flag-144": 1.3,
    "flag-231": 0.7,
}


def deployment_score(rolled_back: frozenset[str]) -> float:
    """Stand-in for a CI run, validation job, or hardware measurement."""

    recovered = sum(HARMFUL.get(item, 0.0) for item in rolled_back)
    return 90.0 + recovered


def main() -> None:
    result = triage(
        ITEMS,
        deployment_score,
        budget=57,
        max_suspects=4,
        screening_pairs=24,
        shortlist_size=8,
        seed=240903,
    )
    print(f"found: {[item.item for item in result.suspects]}")
    print(f"truth: {sorted(HARMFUL)}")
    print(f"calls: {result.calls} instead of {len(ITEMS) + 1} exhaustive calls")
    print(f"screen p-value: {result.screen_p_value:.4g}")


if __name__ == "__main__":
    main()
