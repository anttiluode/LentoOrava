# PulseTriage repeated-regression cache gate

`RajoitustenHierarkia` ended its synthetic memory ladder at a useful boundary: expected consequences can save measurements, but simple recency/TTL attackers remove most of the need for clever memory management.

This gate asks whether that mechanism survives in the practical PulseTriage workload.

## Question

> Across repeated regressions in the frozen handwritten-digits classifier, can remembered **healthy rollback outcomes** reduce cumulative validation calls without materially reducing fault recovery?

The existing PulseTriage receipt already uses a real classifier and a scalar-only rollback API. This gate keeps that classifier and adds one production-like nuisance: rolling back a healthy change has a small expected KPI side-effect. Those expected side-effects vary with validation-traffic shard mix over time.

A regression is therefore not simply “rollback makes the number better.” It is:

```text
observed rollback effect
        -
expected healthy rollback effect
        =
excess consequence caused by the regression
```

The expected consequence may be cached across incidents.

## Workload

- same frozen scikit-learn digits classifier as the executed PulseTriage receipt;
- 256 candidate pixel/shard preprocessing changes;
- four true faults per incident, sampled from the same preregistered informative pool;
- 24 fixed balanced group rollback masks;
- 24 incidents per sequence;
- validation shard mix follows stable → drift → new-stable epochs;
- public observation remains one aggregate scalar KPI per rollback.

The healthy rollback nuisance is deterministic and fixed before outcomes. Its magnitude is small relative to a typical true-fault repair, but a 128-item grouped rollback accumulates enough nuisance to challenge an uncalibrated screen.

## Cached screen

During a healthy interval, buy the baseline plus the 24 group rollback outcomes and store each expected intervention effect:

```text
E_k = KPI_healthy(mask_k) - KPI_healthy(no rollback)
```

During an incident, buy the current baseline and the same 24 group outcomes:

```text
R_k = [KPI_incident(mask_k) - KPI_incident(no rollback)] - E_k
```

Decode the sparse fault support from the 24 residuals with the same small OMP machinery used by PulseTriage.

The screen deliberately uses **one current measurement per group**, not complementary pairs. The remembered healthy effect supplies the missing reference.

## Attackers

- **ordinary PulseTriage** — no cross-incident memory; 57 current validation calls per incident;
- **uncalibrated one-sided screen** — same 25 current calls as the cached method but subtracts no healthy reference;
- **cache once** — acquire 25 healthy calibration calls at the start and reuse forever;
- **TTL-4 cache** — refresh the 25-call healthy calibration every four incidents;
- **fresh reference** — reacquire healthy calibration before every incident; expensive ceiling for this one-sided design.

If the uncalibrated screen matches the cached screen, memory has not earned work. If ordinary PulseTriage remains preferable after cumulative calibration cost is counted, keep ordinary PulseTriage.

## Product metrics

- recovered lost KPI;
- actual fault recall;
- exact fault-set rate;
- cumulative healthy-calibration calls;
- cumulative incident validation calls;
- total calls per incident;
- total calls per recovered KPI point;
- performance before / during / after traffic-mix drift.

## Locked useful boundary

The cache is product-shaped only if TTL-4:

1. recovers at least **85%** of lost KPI on average;
2. recalls at least **75%** of true faults;
3. stays within **5 percentage points** of ordinary PulseTriage recovery;
4. uses at most **70%** as many cumulative scalar calls as ordinary PulseTriage.

These are not claims of optimal thresholds. They are a kill boundary for whether the cross-repo memory mechanism is worth carrying into PulseTriage at all.

## Claim boundary

This is still a controlled classifier regression workload, not a production deployment. A positive result would earn only:

> **reusing healthy black-box intervention outcomes can amortize repeated scalar-only regression diagnosis.**

A negative result means the memory layer should remain in the research lineage and stay out of the practical PulseTriage story.
