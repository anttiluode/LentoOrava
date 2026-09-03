# LentoOrava — bounded observers make one image

**Flying squirrel conservation program. Sol thinking repo.**

`LentoOrava` is the first concrete image-model build of a recurring idea from the Geometric Neuron / V24 / Child / BlackBoxLab / Operaattori line:

> **The global state does not need to be globally observed. It only needs to be causally shared.**

A normal image model is allowed to process the whole latent tensor at once. LentoOrava deliberately inserts a harsher interface. A population of small observer-writers gets only a local patch, its current address, and a private recurrent state. Each observer may move and write locally into one shared latent field. A tiny pointwise decoder reads the final field as RGB.

The first question is not whether this beats diffusion. It is whether **global image structure can be assembled by bounded local agents at all**, and whether active addressing earns anything over matched boring controls.

## The machine

```text
partially observed image
        |
        v
pointwise input projection
        |
        v
+---------------- shared latent field X_t ----------------+
|                                                          |
|   local transport operator                               |
|           |                                              |
|       +---+-----------+-----------+-----------+          |
|       |               |           |           |          |
|   observer 1      observer 2   ... observer N            |
|   local READ      local READ       local READ             |
|   private h_1     private h_2      private h_N            |
|   choose move     choose move      choose move            |
|   local WRITE     local WRITE      local WRITE             |
|       +---------------+-----------+-----------+            |
|                       |                                ^   |
+-----------------------+--------------------------------+---+
                        |
                 pointwise RGB decoder
                        |
                        v
                    one image
```

No observer receives global pooling, a flattened image, or global attention. Its observation is

```text
(local patch, current address, private state)
```

and its action is

```text
(move address, local Gaussian write)
```

The shared field is the only place where their work becomes global.

## Gate 0 — long-range copy through a shared body

The synthetic target contains **two same-colour Gaussian blobs** at the same vertical coordinate. The input exposes only the left blob; the right blob is absent.

A pixel-local solution cannot know the missing right-hand colour or y-coordinate. Evidence has to travel through state.

Three arms use the same basic latent field:

| arm | local agents | private recurrent state | movable address | local write |
|---|---:|---:|---:|---:|
| **ACTIVE** | yes | yes | **yes** | yes |
| **FIXED** | yes | yes | no | yes |
| **TRANSPORT** | no | no | no | no |

The main metric is MSE on the hidden right half, not merely whole-image MSE.

The important outcome is not "ACTIVE wins" by definition. A clean negative is useful:

- if `TRANSPORT` matches it, the agents are decoration;
- if `FIXED` matches it, active addressing is unnecessary;
- if all fail, bounded local writes are insufficient in this implementation;
- if `ACTIVE` wins on held-out y positions/colours, then address choice has earned a role.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q

python train.py --variant active --epochs 12
python train.py --variant fixed --epochs 12
python train.py --variant transport --epochs 12
```

Or run the three-arm comparison:

```bash
python experiments/gate0_compare.py --epochs 12 --device cpu
```

On CUDA, use `--device cuda`.

Make a trajectory panel from a trained ACTIVE checkpoint:

```bash
python demo.py results/active_seed0.pt
```

The fourth panel plots every observer's address through time. The paths are not an attention visualization computed after the fact; they are the actual coordinates used to sample and write the shared field.

## Model equations

Let the shared latent be

```text
X_t in R^(C x H x W)
```

and observer `i` have address `a_i(t)` and private state `h_i(t)`.

The shared body first performs a local transport step

```text
X'_t = X_t + alpha A_theta(X_t)
```

where `A_theta` is a depthwise-local learned operator.

Each observer then samples only a small patch

```text
r_i(t) = READ(X'_t, a_i(t))
```

updates private history

```text
h_i(t+1) = GRU(h_i(t), r_i(t), a_i(t))
```

moves by a bounded amount

```text
a_i(t+1) = clip(a_i(t) + delta_i(h_i), -1, 1)
```

and makes a local differentiable Gaussian write

```text
X_(t+1) = X'_t + sum_i gate_i * Gaussian(a_i) * value_i.
```

Finally a **1x1 decoder** maps the field to RGB. The decoder therefore cannot create long-range structure by itself; spatial communication must already be present in the field.

## Why this is related to diffusion only at a high level

Diffusion models demonstrate the uncontroversial fact that iterative transformations of a shared latent can end in a globally coherent image. LentoOrava is testing a different computational constraint:

> can that latent be organized when the workers themselves are bounded observers with local read/write access rather than globally applied blocks?

This repository makes no claim of superiority to diffusion, CNNs, transformers, neural cellular automata, active vision, or recurrent attention. Those are attackers and neighboring literatures, not rhetorical enemies.

## Lineage

The machinery is a convergence, not a claim that the old metaphors were all correct:

- **Geometric-Neuron** — history represented through geometry; early structural adaptation experiments.
- **GeometricNeuronV24** — address + scalar measurement; active READ/WRITE; observability rather than fixed readout.
- **Child** — if a prediction changes the state from which the next prediction is made, prediction becomes dynamics.
- **BlackBoxLab** — sampling policy can drive specialization, but can also collapse into monoculture.
- **Twensday** — persistent structure is a constrained parameterization of an effective operator.
- **Operaattori / Jako** — separate transport from nonlinear state-mediated response.
- **TinyAvatar / SplatWorld** — an image as the final readout of a shared latent/field is already a useful implementation language.

LentoOrava keeps only the executable seam:

```text
bounded observation
      +
private history
      +
local action
      +
causally shared state
      =
possible global output
```

## Next gates if Gate 0 survives

1. **Gate 1 — relocation:** move the hidden partner relation and test whether ACTIVE learned a policy rather than one address.
2. **Gate 2 — multiscale:** let the same agent choose patch scale as well as position; compare information-gain-like policies to learned movement.
3. **Gate 3 — specialization:** remove identical initial roles and measure whether observers differentiate into stable functions.
4. **Gate 4 — operator memory:** let repeated local traffic slowly alter transport itself, then test whether history changes future image completion after fast state is erased.
5. **Gate 5 — real images:** masked CIFAR/CelebA inpainting with matched CNN/recurrent-attention/neural-CA attackers.
6. **Only then:** conditional generation from noise and text.

The rule for the repo is the same as the newer work elsewhere: **attackers first, claims second.**
