import numpy as np

from lentoorava.pulse_repair import (
    Region,
    ScalarPulseOracle,
    adaptive_pulse_search,
    apply_selected_repairs,
    apply_wounds,
    make_symmetric_body,
    mirrored_trial,
    random_pulse_search,
    recovered_fraction,
)


def test_mirror_write_is_local_and_exact_on_symmetric_target():
    target = make_symmetric_body(48)
    damaged = apply_wounds(target, [(0.55, 0.1, 0.16)])
    region = Region(32, 20, 39, 27)
    repaired = mirrored_trial(damaged, region)

    outside = np.ones(damaged.shape[:2], dtype=bool)
    outside[region.y0 : region.y1, region.x0 : region.x1] = False
    assert np.array_equal(repaired[outside], damaged[outside])
    assert np.allclose(
        repaired[region.y0 : region.y1, region.x0 : region.x1],
        target[region.y0 : region.y1, region.x0 : region.x1],
        atol=1e-6,
    )


def test_probes_are_reversible_and_respect_budget():
    target = make_symmetric_body(48)
    damaged = apply_wounds(target, [(0.6, -0.1, 0.18)])
    before = damaged.copy()
    oracle = ScalarPulseOracle(target)
    result = adaptive_pulse_search(
        damaged,
        oracle,
        budget=24,
        repair_slots=3,
        tile_size=4,
    )
    assert np.array_equal(damaged, before)
    assert result.pulse_calls <= 24
    assert len(result.selected) <= 3
    assert all(p.region.x0 >= damaged.shape[1] // 2 for p in result.selected)


def test_adaptive_search_beats_equal_budget_random_on_fixed_suite():
    target = make_symmetric_body(96)
    active_scores = []
    random_scores = []
    for seed in range(32):
        rng = np.random.default_rng(seed)
        wounds = [
            (
                float(rng.uniform(0.28, 0.80)),
                float(rng.uniform(-0.55, 0.55)),
                float(rng.uniform(0.11, 0.18)),
            )
            for _ in range(int(rng.integers(1, 4)))
        ]
        damaged = apply_wounds(target, wounds)

        active_oracle = ScalarPulseOracle(target)
        active = adaptive_pulse_search(
            damaged,
            active_oracle,
            budget=36,
            repair_slots=5,
            tile_size=6,
        )
        active_scores.append(
            recovered_fraction(
                active_oracle,
                damaged,
                apply_selected_repairs(damaged, active.selected),
            )
        )

        random_oracle = ScalarPulseOracle(target)
        random = random_pulse_search(
            damaged,
            random_oracle,
            budget=36,
            repair_slots=5,
            tile_size=6,
            rng=np.random.default_rng(seed + 10_000),
        )
        random_scores.append(
            recovered_fraction(
                random_oracle,
                damaged,
                apply_selected_repairs(damaged, random.selected),
            )
        )

    assert np.mean(active_scores) > np.mean(random_scores) + 0.30
    assert np.mean(np.asarray(active_scores) > np.asarray(random_scores)) >= 0.80

