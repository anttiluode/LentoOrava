# Gate 0 diagnosis — the first task was not yet being solved

The first full 12-epoch run was useful because it exposed two separate assay problems.

## 1. ACTIVE beat the architecture ablations, but not the trivial data distribution

Hidden-right MSE at the end of the original run:

| arm | hidden-right MSE |
|---|---:|
| ACTIVE | 0.007609 |
| FIXED | 0.008155 |
| TRANSPORT | 0.008480 |

ACTIVE's best epoch was epoch 11 at 0.007587. That is a real ACTIVE > FIXED > TRANSPORT ordering, but it is not enough.

Exact boring attackers obtained by enumerating the finite y x colour support:

| attacker | hidden-right MSE |
|---|---:|
| zeros | 0.009117 |
| noisy input / identity | 0.008964 |
| **dataset mean** | **0.007520** |
| knows colour, averages over y | 0.006737 |
| knows y, averages over colour | 0.003000 |
| oracle y + colour | 0 |

So the old classification is:

`ACTIVE_BEATS_ARCHITECTURE_ABLATIONS_BUT_NOT_TRIVIAL_DATASET_MEAN`

The broad vertical smear in the demo is consistent with an uncertainty-average solution rather than a copied y coordinate.

## 2. The old start grid often failed to acquire the fact in the first place

The source blob sits at x=7 and y ranges from 5 through 26. Under the original 12-agent square start grid, the nearest initial observer is:

```text
minimum distance to source centre    3.93 px
median distance                      5.24 px
maximum distance                     7.07 px
```

But the local READ window is only 3 x 3. Gate 0 was therefore confounding two questions:

1. can a bounded observer acquire the local fact?
2. once acquired, can that fact cause a corresponding distant write?

Those must be separated.

## Gate 0A repair

Observers now start on a vertical source rail spanning the allowed y range.

A new CARRIER arm uses the same local READ, private recurrent state, latent write and 1x1 decoder, but its x address is scripted to move from source to target while preserving y.

Interpretation:

```text
CARRIER fails
    -> the read/carry/write body or loss is broken

CARRIER passes, ACTIVE fails
    -> the learned movement policy is the bottleneck

ACTIVE passes, FIXED fails
    -> learned mobility has earned a causal role
```

Training also receives a coarse-to-fine Gaussian hidden-half loss at scales 6, 3, 1.5 and 0 px plus a weak mass term.

Finally, `experiments/gate0_policy_audit.py` measures whether actual trajectories change when y, colour, or the presence of evidence changes.

Do not call Gate 0 passed merely because ACTIVE beats FIXED. First require CARRIER to beat the dataset-mean attacker, then require ACTIVE to beat the boring baselines and show evidence-dependent trajectories.
