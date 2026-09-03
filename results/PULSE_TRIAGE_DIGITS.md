# PulseTriage — real classifier rollback receipt

## Question

Can coded reversible rollbacks turn one aggregate model-quality number into a
useful regression shortlist with materially fewer validation runs than testing
every candidate change?

## Locked gate

Before the 64-trial receipt, the practical gate was fixed at:

- at least 85% mean lost-KPI recovery;
- at least 80% mean true-fault recall;
- at least 45 percentage points more recovery than equal-budget random rollback;
- no more than 25% of exhaustive individual calls;
- no more than 40% recovery after pulse addresses are shuffled.

## Workload

- scikit-learn handwritten digits, 8×8 pixels;
- stratified 65/35 train/validation split, seed `42`;
- frozen `StandardScaler + LogisticRegression` classifier;
- clean validation accuracy: **96.98%**;
- 256 named candidate changes: 64 pixel transforms × four validation shards;
- four broken preprocessing changes per trial;
- faults sampled uniformly from a declared fixed pool: the 24 strongest
  coefficient-norm pixel features × four shards;
- legal intervention: temporarily roll back any addressed subset;
- only public observation: aggregate validation negative log loss;
- 64 deterministic held-out fault sets, seed `260903`.

Faults are intentionally sampled from influential features because the assay is
about localizing an observed regression, not detecting changes that have no
measurable consequence.

## Result

| method | mean KPI recovered | mean true-fault recall | calls |
|---|---:|---:|---:|
| **PulseTriage** | **92.77%** | **87.11%** | **57** |
| equal-budget random individual rollback | 20.81% | 17.58% | 57 |
| exhaustive individual rollback | 98.39% | 92.97% | 257 |
| shuffled pulse address | 0.00% | 0.00% | 57 |
| true injected-fault rollback | 100% | 100% | unavailable oracle |

PulseTriage found the complete injected fault set in **68.75%** of trials and
returned a conclusive status in **95.31%**. Median permutation-calibrated screen
`p` was **0.0154**.

Operationally, it obtained **94.29% of exhaustive individual rollback's KPI
recovery with 22.18% of its scalar evaluations**. Equal-budget random search
lost by 71.96 percentage points of recovered KPI.

All five locked checks passed:

> **`PULSE_TRIAGE_PASSES_REAL_CLASSIFIER_ROLLBACK_GATE`**

Machine-readable receipt: [`PULSE_TRIAGE_DIGITS.json`](PULSE_TRIAGE_DIGITS.json).

Reproduce:

```bash
python -m pip install -r requirements-pulsetriage.txt
python -m pip install -e . --no-deps
python experiments/pulse_triage_digits.py \
  --trials 64 \
  --permutations 64 \
  --out results/PULSE_TRIAGE_DIGITS.json
```

## Scope and kill conditions

This is a real classifier and real aggregate loss, but the preprocessing faults
are controlled injections. The experiment establishes a reusable query-budget
result, not automatic diagnosis of arbitrary production failures.

The tool should be rejected or return `inconclusive` when:

- rollback effects are dominated by high-order interactions;
- the evaluator cannot make clean reversible trials;
- faults are dense rather than sparse;
- measurement drift overwhelms the effect;
- direct single-candidate confirmations do not improve the baseline.

The address-shuffle attacker is the key causal control: preserving the number
and size of scalar evaluations while severing their link to requested rollbacks
reduced recovery from 92.77% to zero.
