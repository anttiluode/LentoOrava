import numpy as np

from pulsetriage import triage


def test_coded_triage_finds_sparse_additive_regressions_under_budget():
    items = [f"change-{index:03d}" for index in range(128)]
    harmful = {"change-007": 1.8, "change-044": 0.7, "change-096": 1.2}

    def score(rolled_back):
        return 20.0 + sum(harmful.get(item, 0.0) for item in rolled_back)

    result = triage(
        items,
        score,
        budget=41,
        max_suspects=3,
        screening_pairs=17,
        shortlist_size=6,
        seed=19,
        permutations=64,
    )

    assert result.calls == 41
    assert result.status == "ok"
    assert {candidate.item for candidate in result.suspects} == set(harmful)
    assert result.screen_p_value <= 1 / 13


def test_lower_is_better_and_direct_confirmation_filter_false_screen_item():
    items = [f"part-{index}" for index in range(64)]
    harmful = {"part-11": 4.0, "part-52": 2.0}

    def latency(rolled_back):
        return 100.0 - sum(harmful.get(item, 0.0) for item in rolled_back)

    result = triage(
        items,
        latency,
        budget=29,
        max_suspects=2,
        screening_pairs=12,
        shortlist_size=4,
        higher_is_better=False,
        seed=2,
        permutations=32,
    )

    assert [candidate.item for candidate in result.suspects] == ["part-11", "part-52"]
    assert all(candidate.confirmed_effect > 0 for candidate in result.suspects)


def test_pulse_address_shuffle_is_reported_as_inconclusive():
    items = [f"item-{index}" for index in range(96)]
    rng = np.random.default_rng(4)

    def addressless_noise(_rolled_back):
        return float(rng.normal())

    result = triage(
        items,
        addressless_noise,
        budget=37,
        max_suspects=3,
        screening_pairs=15,
        shortlist_size=6,
        baseline_repeats=1,
        min_effect=10.0,
        seed=7,
        permutations=64,
    )

    assert result.status == "inconclusive"
    assert not result.suspects
    assert result.calls == 37


def test_pure_interaction_without_single_candidate_effect_is_not_actioned():
    items = [f"toggle-{index}" for index in range(64)]
    interacting_pair = {"toggle-9", "toggle-41"}

    def pair_only_score(rolled_back):
        return float(interacting_pair <= set(rolled_back))

    result = triage(
        items,
        pair_only_score,
        budget=37,
        max_suspects=2,
        screening_pairs=15,
        shortlist_size=6,
        seed=23,
        permutations=64,
    )

    assert result.status == "inconclusive"
    assert not result.suspects
    assert all(candidate.confirmed_effect == 0 for candidate in result.shortlist)
