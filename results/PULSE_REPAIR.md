# Pulse Repair — scalar-only fault-localization receipt

## Question

Can adaptive addressed perturbations recover enough spatial causal information
from one scalar consequence channel to send scarce local repair writes to compact
damage?

## Assay

- 512 deterministic trials; seed `240903`.
- Procedural 96×96 bilateral body.
- One to three unseen soft circular wounds on the right half.
- Valid local action: copy a region from its intact mirror location.
- Search never receives the target, wound mask, pixel loss, or gradient.
- One reversible trial write returns one scalar change in whole-body squared error.
- 128 possible 6×6 local destination tiles.
- 36 scalar-call budget, including the cached baseline.
- Five committed local writes.

The adaptive arm recursively bisects the most causally promising region. The
random attacker measures uniformly selected local tiles and retains its best
five under the same scalar-call budget. The exhaustive ruler measures all 128
tiles (129 calls including its baseline) and also retains its best five.

## Result

| method | mean damage recovered | median | mean scalar calls |
|---|---:|---:|---:|
| **adaptive pulse search** | **81.09%** | 86.86% | 34.5 |
| equal-budget random | 26.28% | 23.41% | 36.0 |
| exhaustive 128-tile scan | 87.06% | 90.53% | 129.0 |

Adaptive beat random on **95.90%** of paired wounds, tied on 1.17%, and recovered
an additional 54.82 percentage points on average.

At 26.74% of the exhaustive scan's scalar calls, adaptive search obtained 93.15%
of its recovered damage.

Machine-readable values: [`PULSE_REPAIR.json`](PULSE_REPAIR.json).

Reproduce:

```bash
python experiments/pulse_repair_benchmark.py \
  --trials 512 \
  --budget 36 \
  --repair-slots 5 \
  --out results/PULSE_REPAIR.json
```

## Scope

This is a controlled sparse-fault assay with a correct bilateral repair
primitive. It does **not** show general image restoration, learned regeneration,
or biological credit assignment. The result isolates a smaller reusable claim:

> When useful local interventions are spatially sparse, hierarchical addressed
> probes can turn a scalar global consequence into an actionable location more
> efficiently than random local probing.

