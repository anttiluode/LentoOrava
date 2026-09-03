# Gate 0 smoke execution

This is **not** a scientific receipt and does not decide the ACTIVE/FIXED/TRANSPORT comparison.

It is a tiny local CPU execution used only to verify that all three training arms run, backpropagate and reduce loss before committing the scaffold.

Configuration:

```text
epochs       2
train        256 synthetic examples
validation    64 examples
channels      12
agents         8
steps          5
seed           0
```

Final validation after epoch 2:

| arm | full MSE | hidden-right MSE |
|---|---:|---:|
| ACTIVE | 0.147887 | 0.146178 |
| FIXED | 0.146743 | 0.148187 |
| TRANSPORT | 0.156909 | 0.157136 |

The tiny run only establishes that the implementation executes and learns something. ACTIVE and FIXED are essentially tied at this scale, so **active addressing has not earned a claim**.

Separately, the local unit suite completed:

```text
3 passed
```

A serious Gate 0 run must use the full registered dataset/epoch budget, multiple seeds, and the hidden-right metric.
