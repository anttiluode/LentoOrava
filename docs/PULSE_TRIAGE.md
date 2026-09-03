# PulseTriage

Find a sparse regression among many reversible candidate changes when each
evaluation returns one scalar score.

Typical candidates are feature flags, configuration edits, data shards,
pipeline stages, model blocks, service instances, or hardware channels. The
tool is worthwhile when a full validation or measurement is expensive and a
group of candidates can be safely rolled back for one trial.

## Thirty-second run

From the repository root:

```bash
python examples/flag_regression.py
```

Expected result:

```text
found: ['flag-017', 'flag-144', 'flag-083', 'flag-231']
truth: ['flag-017', 'flag-083', 'flag-144', 'flag-231']
calls: 57 instead of 257 exhaustive calls
```

The order is effect size, not identifier order.

## Python API

```python
from pulsetriage import triage


def score(rolled_back: frozenset[str]) -> float:
    # Build a temporary configuration with exactly these candidates reverted.
    # Run the real validation, canary, simulation, or hardware measurement.
    # Return one finite number. Larger is better in this example.
    return run_validation(rolled_back)


result = triage(
    ["flag-a", "flag-b", "flag-c", "flag-d"],
    score,
    budget=15,
    max_suspects=1,
)

if result.status == "ok":
    for suspect in result.suspects:
        print(suspect.item, suspect.confirmed_effect)
else:
    print(result.warnings)
```

Set `higher_is_better=False` for error, latency, cost, or other metrics that
should decrease. Use `baseline_repeats`, `screen_repeats`, and
`confirm_repeats` when the evaluator is noisy; they count against `budget`.
Set a domain-meaningful `min_effect` when tiny statistically real changes are
not operationally useful.

## Command line

Install the local package:

```bash
python -m pip install -e .
```

Put one candidate identifier on each line of `changes.txt`. Then run:

```bash
pulsetriage \
  --items changes.txt \
  --budget 57 \
  --max-suspects 4 \
  --output triage.json \
  -- python your_evaluator.py
```

For every evaluation the command receives a JSON array in the environment
variable:

```text
PULSE_TRIAGE_ROLLBACK_JSON
```

The command must print either a number or a final JSON line such as:

```json
{"score": 0.9137}
```

Each invocation must start from the same deployed baseline, apply only the
requested temporary rollbacks, measure, and clean up. PulseTriage does not
perform deployment or rollback itself.

## What it does

With candidate rollback effects `w_i`, an approximately additive score obeys:

```text
score(S) ≈ score(empty) + sum(w_i for i in S)
```

PulseTriage asks paired, balanced questions. One evaluation rolls back the `+`
half and another rolls back the complementary `-` half:

```text
d_q = score(S_q+) - score(S_q-) ≈ a_q · w
```

The resulting signed measurement matrix is decoded with a small orthogonal
matching-pursuit implementation. This produces only a shortlist. Every returned
suspect must then improve the baseline in a direct one-candidate rollback.

Because sparse decoders can fit noise surprisingly well when there are more
candidates than measurements, the tool also shuffles pulse outcomes against
their rollback addresses and repeats the sparse fit. `screen_p_value` reports
how often the address-shuffled fits look at least as good. A weak screen or no
directly confirmed improvement makes the status `inconclusive`.

## When to use it

Use PulseTriage when:

- there are many named candidate changes;
- only a few are expected to cause most of the regression;
- several candidates can be rolled back together without invalidating a test;
- the same scalar KPI can be measured for every temporary configuration;
- evaluations are expensive enough that 57 runs instead of 257 matters.

Do not use it when:

- exhaustive individual evaluation is already cheap;
- rollbacks are destructive, unsafe, or stateful;
- the failure requires a precise combination but no member helps alone;
- most candidates have large mixed positive and negative effects;
- uncontrolled drift is larger than the rollback effects;
- the score cannot be reproduced against a stable baseline.

In those cases, the tool should either be given repeats and a real effect
threshold or be left unused. `inconclusive` is a result.

## Relation to known methods

This is not a new optimization theory. It sits between:

- [delta debugging](https://www.st.cs.uni-saarland.de/dd/), which minimizes a
  failure-inducing input or change set using automated pass/fail experiments;
- [adaptive group testing](https://arxiv.org/abs/2106.12193), which identifies a
  sparse defective set using pooled tests;
- sparse recovery and ordinary reversible ablation.

The small contribution here is an auditable software seam for **quantitative
scalar KPIs**: complementary rollback masks, sparse screening, direct
confirmation, a strict call ledger, and an address-shuffle refusal test.

## Reproduce the real-workload receipt

```bash
python -m pip install -r requirements-pulsetriage.txt
python experiments/pulse_triage_digits.py \
  --trials 64 \
  --out results/PULSE_TRIAGE_DIGITS.json
```

The benchmark trains a frozen logistic classifier on scikit-learn's 8×8 digits
dataset, injects four preprocessing-shard regressions among 256 candidates, and
compares PulseTriage with equal-budget random individual rollback, exhaustive
individual rollback, the true-fault ceiling, and shuffled pulse addresses.
