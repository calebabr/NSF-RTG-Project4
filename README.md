# NSF RTG Project 4: Bias Collapse in Shallow ReLU Networks

## Overview

This project numerically investigates three open problems from the course slides:
**Open Problem 4.1** (cluster count conjecture), **Open Problem 4.2** (higher-dimensional
collapse), and **Open Problem 4.3** (provable pruning bound).

The 1D setting (Problems 4.1 and 4.3) studies shallow ReLU networks trained by gradient
flow on smooth target functions. The central observed phenomenon is that the bias parameters
spontaneously collapse into tight clusters over training, and the number of clusters appears
to converge toward k, the number of inflection points of the target function, as the network
width m grows.

The 2D setting (Problem 4.2) asks whether an analogous collapse generalizes to higher
dimensions. In d=2, each neuron defines an oriented line (hyperplane) parameterized by an
angle θ and a normalized offset β. The question is whether gradient flow concentrates the
neuron measure onto a lower-dimensional set in this (θ, β) space, and whether the collapse
mode is determined by the structure of the target function.

---

## Coverage of the Open Problems

### Open Problem 4.1: Cluster Count

**Conjecture from the slides:**

$$\lim_{m \to \infty} C(m, f^\ast) = \lvert\{x \in [-1,1] : (f^\ast)''(x) = 0 \text{ and changes sign}\}\rvert$$

| Specific Goal | Description | Status |
|---|---|---|
| 4.1.1 | Prove C(m, f*) ≤ Cmax(f*) independent of m | Numerically supported via simulate_parallel.py and instability_test.py; experiment in process |
| 4.1.2 | Is convergence finite-time or asymptotic? | Data implies asymptotic (C decreases gradually with T); not yet explicitly addressed |
| 4.1.3 | Does count depend on curvature amplitude or only sign pattern of f''? | Not addressed; would require comparing targets with identical inflection locations but different curvature magnitudes |

#### Current Numerical Findings (T=500, m up to 3500; m=5000 in progress)

69 runs across 9 targets. All runs verified stationary (max|da/dt| < 0.01 for all but 3).
C(m, f*) follows a non-monotone rise-then-fall pattern universally; the peak shifts to
larger m as k increases.

| Target | k | C(m) at m = 50, 100, 250, 500, 1000, 1500, 2000, 3500 |
|---|---|---|
| sin_1pi | 1 | 31, 45, 22, 6, **1, 1, 1, 1** |
| sin_2pi | 3 | 26, 29, 13, 8, **2**, †, †, † |
| sin_3pi | 5 | 11, 16, 23, 14, **5**, 2, †, † |
| sin_4pi | 7 | 9, 11, 21, 26, 17, **7**, 2, † |
| sin_5pi | 9 | 11, 11, 18, 23, 31, 13, 2, — |
| sin_6pi | 11 | 13, 14, 14, 15, 31, 40, 13, — |
| sin_7pi | 13 | 14, 15, 18, 17, 22, 36, 34, — |
| poly_k3 | 3 | 29, 45, 35, 20, 11, 8, 7, 4 |
| x_cubed | 1 | 31, 40, 31, 16, 13, 6, 7, 4 |

†**Dense-packing artifact:** gap-based counter collapses to C=1 when biases are too dense
to detect gaps (> 0.02). Affects sin_2pi m ≥ 1500, sin_3pi m ≥ 2000, sin_4pi m=3500.
Excluded from limit conclusions.

**Conclusions and next steps:**

- **Conjecture supported for k=1 (sin_1pi):** C stabilizes at 1 from m=1000 through m=3500.
- **k=5, 7:** C passes through k then falls below it — not convergence in the limit sense.
  The critical open question is whether the below-k values are true stationary states or
  initialization-dependent local minima. Running multiple seeds at the same (m, T) would
  resolve this.
- **k=9, 11, 13:** Still in the rise phase; need m well beyond 3500.
- **Polynomial targets (poly_k3, x_cubed):** Converge far more slowly than trigonometric
  targets with the same k — inflection point geometry matters beyond just counting k.
- **Blocking issue:** The fixed gap threshold (0.02) must be replaced with an adaptive
  threshold before reliable conclusions can be drawn for m ≥ 1500.

### Open Problem 4.2: Higher-Dimensional Collapse

**Conjecture from the slides:** For structured d-dimensional targets, gradient flow
concentrates the neuron measure onto a lower-dimensional set in hyperplane space. The
support may be points (same hyperplane), curves (parallel offsets), or several direction
families.

| Specific Goal | Description | Status |
|---|---|---|
| 4.2.1 | Does directional collapse occur for structured targets? | Partially confirmed: angular entropy drops for ridge and separable targets, especially at small width |
| 4.2.2 | Does collapse mode track target structure? | Confirmed at m=64: ridge shows mild directional concentration, separable shows family splitting, radial resists collapse |
| 4.2.3 | Does collapse generalize to larger width? | Inverted: collapse is strongest at small m and weakens as m grows; large-m behavior differs from theoretical prediction |
| 4.2.4 | Are collapsed neurons functionally redundant (prunable)? | Not confirmed; pruning diagnostic is unreliable due to sign-cancellation in output weights |

### Open Problem 4.3: Provable Pruning

**Bound from the slides:**

$$\|\tilde{f} - f\|_{L^2} \leq \delta \cdot \sum_{j=1}^{m} |a_j|$$

| Aspect | Status |
|---|---|
| Numerically verify the bound holds | Confirmed by verify_pruning.py across all completed runs |
| Bound the key challenge: Σ\|aⱼ\| grows with m | Tracked in summary figure; growth observed but not controlled; remains open |

---

## Research Goals and Script Mapping

### Goal 1: Verify the Cluster Count Conjecture (4.1 conjecture, 4.1.1)

Numerically confirm that C(m, f*) converges toward k as m grows, and that this convergence
stabilizes independently of how large m gets once past a threshold. Addressed by
**simulate_parallel.py**.

### Goal 2: Show Configurations Near k Are Unstable Above k and Attracted to k (4.1.1)

Test the stability of configurations where the cluster count is within a threshold of k.
Three run categories: **exact_k**, **above_k**, and **below_k** (threshold = 2). Addressed
by **instability_test.py**. Supports 4.1.1 by building evidence that k acts as an attractor
from both sides.

### Goal 3: Verify Stationary Point Conditions (4.1 conjecture support)

At convergence, verify ODE velocities are approximately zero and the integrated residual R_j
from each bias to 1 is approximately zero for every active neuron. Addressed automatically
inside **simulate_parallel.py**.

### Goal 4: Numerically Verify the Pruning Bound (4.3)

Verify ||f_tilde minus f|| ≤ δ · Σ|aⱼ| across all converged runs, and track how the bound
components behave as m grows. Addressed by **verify_pruning.py**.

### Goal 5: Test Whether Collapse Generalizes to d=2 (4.2)

Train a shallow ReLU network on three structurally distinct 2D target functions and measure
whether gradient descent concentrates neuron directions and offsets into clusters over
training. The three targets are chosen to provoke different theoretically predicted collapse
modes: directional collapse (ridge), family splitting (separable), and no collapse (radial).
Addressed by **higher_dim_collapse.py**.

### Goal 6: Build Intuition Toward a Formal Proof (4.1 and 4.3)

Synthesize numerical evidence from all goals to identify proof strategies. This is an
interpretive goal informed by all scripts.

---

## Project Structure

```
MathProject4/
|
|-- MathProject Slides.pdf          Source slides: model, ODEs, and open problems
|
|-- simulate_parallel.py            Main simulation — m sweep at T=500, all 9 targets (Goals 1, 3)
|-- verify_pruning.py               Pruning bound verification (Goal 4)
|-- instability_test.py             k+1 instability test (Goal 2)
|-- higher_dim_collapse.py          2D collapse experiment (Goal 5)
|-- regenerate_figures.py           Helper: regenerates clean final fit figures
|
|-- archive/                        Superseded scripts (simulate_discrete.py data still used)
|   |-- simulate_discrete.py        Discrete GD version — 52 runs complete, data in figures/
|
|-- figures/
|   |-- Replication data/
|       |-- {target}/
|       |   |-- m={m}/
|       |       |-- T={T}/
|       |           |-- slide93_reproduction.png      Bias trajectories, final fit, loss
|       |           |-- clusters_vs_inflections.png   Cluster locations vs inflection pts
|       |           |-- ode_verification.png          ODE velocities and R_j at convergence
|       |           |-- convergence_check.csv         Per neuron da, db, R_j, active flag
|       |           |-- run_meta.csv                  Single row summary for restart safety
|       |           |-- pruning_verification.png      Pruning bound check per run
|       |           |-- goal2_near.png                k+1 instability, near injection
|       |           |-- goal2_isolated.png            k+1 instability, isolated injection
|       |
|       |-- run_summary_parallel.csv         Results from simulate_parallel.py
|       |-- convergence_plot_parallel.png    C(m) vs m for all 9 targets
|
|   |-- Discrete GD/
|       |-- {target}/m={m}/steps={n}/        Per-run figures and CSVs
|       |-- convergence_plot_current.png     C(m) vs m for discrete GD
|       |-- pruning_bound_results.csv        Pruning bound metrics across all runs
|       |-- pruning_bound_summary.png        Summary plots for Open Problem 4.3
|       |-- goal2_results.csv               Instability test results across all runs
|       |-- goal2_summary.png               Summary heatmap and return rates
|
|   |-- collapse_v2_plots_and_results/
|       |-- collapse_v2_results.json        Diagnostic snapshots across all runs
|       |-- collapse_v2_summary.png         3x3 summary grid: all targets x all widths
|       |-- collapse_v2_ridge.png           Per-target diagnostic plot (ridge)
|       |-- collapse_v2_separable.png       Per-target diagnostic plot (separable)
|       |-- collapse_v2_radial.png          Per-target diagnostic plot (radial)
```

---

## Scripts

### simulate_parallel.py: Main Simulation

**Open problems addressed:** 4.1 (main conjecture), 4.1.1 (partial), Goal 3

**Purpose:** Extends the m sweep to m=5000 at T=500, across all 9 targets including the
three new high-k targets (k=9, 11, 13). All runs use T=500 — long enough to reach
stationarity at all tested m values (verified: max|da/dt| < 0.01 for nearly all runs).

**Targets and parameters:**

| Targets | m values | T |
|---|---|---|
| All 9 (sin_1pi through sin_7pi, x_cubed, poly_k3) | 50, 100, 250, 500, 1000, 1500, 2000, 3500, 5000 | 500 |

**Speed optimization:** N_QUAD reduced from 400 to 200. The quadrature grid evaluates
spatial integrals at every ODE step; halving its size gives approximately 1.9x speedup
with no change to cluster counts at m ≥ 100.

**Convergence plot:** C(m, f*) vs m at T=500 for all 9 targets.

**Outputs per run:** slide93_reproduction.png, clusters_vs_inflections.png,
ode_verification.png, convergence_check.csv, run_meta.csv

**Global outputs:** run_summary_parallel.csv, convergence_plot_parallel.png

**Run with:** `python simulate_parallel.py`

---

### verify_pruning.py: Open Problem 4.3 Verification

**Open problems addressed:** 4.3 (pruning bound)

**Purpose:** Reads saved b_j and a_j values from each run folder and verifies the pruning
bound. No re-simulation needed.

**Current result:** Bound holds for all 108 completed runs. Tightness ratios range from
0.001 to 0.15. The amplitude sum Σ|aⱼ| grows with m — proving the bound is non-vacuous
requires showing this growth is controlled relative to how fast delta shrinks.

**Outputs per run:** pruning_verification.png

**Global outputs:** pruning_bound_results.csv, pruning_bound_summary.png

**Run with:** `python verify_pruning.py`

---

### instability_test.py: Goal 2 Instability Test

**Open problems addressed:** 4.1.1

**Purpose:** Injects one extra neuron into converged k-cluster states and tests whether
gradient flow drives the system back to k clusters. Two injection strategies per run: near
(just outside an existing cluster) and isolated (maximally distant from all cluster centers).

**Qualifying runs from current data:** 20 runs across exact_k, above_k, and below_k
categories. More will qualify once simulate_parallel.py completes.

**Outputs per run:** goal2_near.png, goal2_isolated.png, goal2_natural.png (above_k only)

**Global outputs:** goal2_results.csv, goal2_summary.png

**Run with:** `python instability_test.py`

---

### collapse_v2.py: Open Problem 4.2 — 2D Collapse Experiment

**Open problems addressed:** 4.2 (higher-dimensional collapse)

**Purpose:** Trains a shallow ReLU network (f(x) = Σ aⱼ σ(wⱼᵀx + bⱼ), x ∈ [-1,1]²) on
three target functions designed to provoke distinct collapse modes, then tracks the angular
distribution of neuron normals and the (θ, β) point cloud over training.

**The canonical reparameterization:** Each neuron is normalized into scale-invariant
coordinates: θⱼ = angle of wⱼ (which direction the neuron is sensitive to), βⱼ = bⱼ/‖wⱼ‖
(signed distance from origin to kink), αⱼ = aⱼ‖wⱼ‖ (scale-absorbed output weight). Two
neurons defining the same hyperplane have identical (θ, β) regardless of raw weight scaling.

**Target functions:**

| Target | Function | Predicted collapse mode |
|---|---|---|
| Ridge | sin(2πuᵀx) + 0.5sin(4πuᵀx), u=(3/5, 4/5) | All normals concentrate toward θ≈53°; one angular cluster |
| Separable | sin(2πx₁) + 0.3sin(2πx₂) | Two direction families at θ≈0° and θ≈90°; unequal sizes due to broken symmetry |
| Radial | cos(π‖x‖) | No preferred direction; angular entropy stays near uniform |

Note: u=(3/5, 4/5) is deliberately non-axis-aligned to distinguish directional collapse
from coordinate-axis effects. The separable target uses asymmetric weights (1.0 vs 0.3)
to break the x₁/x₂ symmetry and produce unequal family sizes.

**Width sweep:** Each target is trained at m = 64, 256, and 512 to test whether collapse
is a large-width or small-width phenomenon.

**Diagnostics:**

| Diagnostic | What it measures |
|---|---|
| Angular entropy H({θⱼ}) | Shannon entropy of the binned angle distribution. Drops toward 0 if directions collapse; near log(36)≈3.58 if spread is uniform |
| DBSCAN cluster count | Number of clusters in the normalized (θ, β) point cloud. Should track the theoretically predicted number of direction families |

Note: a pruning diagnostic (effective width) was also implemented but found to be
unreliable due to sign-cancellation between neurons with opposite-sign output weights.
Pruning results are excluded from the analysis.

**Key findings:**

1. Collapse mode tracks target structure at small width (m=64): ridge shows mild angular
   concentration, separable shows family splitting with entropy dropping from 3.08 to 2.65,
   radial shows mild collapse due to capacity constraints.

2. Collapse weakens as width increases. At m=512, entropy changes are smaller for all three
   targets. This is the opposite of the theoretical large-m prediction and suggests collapse
   is capacity-constrained rather than asymptotic.

3. Radial at large width (m=512) maintains near-uniform entropy (≈3.54 throughout, against
   a maximum of 3.58), confirming the rotational symmetry prediction only when the network
   is wide enough to cover all orientations.

4. Separable at m=512 stabilizes at 3 DBSCAN clusters from epoch 2700 onward — consistent
   with two direction families plus a noise cluster, the clearest structural result in the
   dataset.

**Parameters:** m ∈ {64, 256, 512}, N_train=4096, N_test=1024, 6000 epochs, Adam lr=1e-3,
snapshots every 300 epochs.

**Outputs:** collapse_v2_results.json (slim diagnostics), collapse_v2_summary.png (3×3
grid: rows = angular entropy / cluster count / prune ratio, cols = targets), and one
per-target diagnostic plot (collapse_v2_{target}.png) with loss curve, entropy trajectory,
(θ,β) point cloud at final epoch, and effective width panel.

**Run with:** `python collapse_v2.py`

---

### simulate_discrete.py (archived): Discrete Gradient Descent

**Purpose:** Tests whether the cluster collapse phenomenon holds under discrete gradient
descent (explicit update steps) rather than the continuous ODE gradient flow studied in
Open Problem 4.1. Uses the same 9 targets and scaled learning rate lr = 0.01/√(m/50)
with steps ∝ √m to maintain comparable total gradient work across m values.

**Key finding:** C(m, f*) does **not** converge to k under discrete GD for any target
except sin_1pi (k=1). Most sequences either collapse below k or stall above it:

| Target | k | C at m=1000 (ODE) | C at m=1000 (Discrete GD) |
|---|---|---|---|
| sin_1pi | 1 | 1 = k | 1 = k |
| sin_3pi | 5 | 5 = k | 4 |
| sin_4pi | 7 | 17 | 1 |
| sin_5pi | 9 | 31 | 5 |
| poly_k3 | 3 | 11 | 8 |
| x_cubed | 1 | 13 | 7 |

All discrete GD runs are verified near-stationary (max gradient < 0.001), so these are
genuine stationary states, not mid-training snapshots.

**Significance:** Open Problem 4.1 is stated for continuous gradient flow. The discrete GD
result clarifies that the conjecture is specific to the continuous-time ODE dynamics —
discretization breaks the convergence. This is not a flaw; it means the ODE structure is
essential to the phenomenon, not incidental.

**Output data:** `figures/Discrete GD/` — 52 runs across 9 targets,
m ∈ {50, 100, 250, 500, 1000, 1500}.

---

### regenerate_figures.py: Figure Cleaner

**Purpose:** Reads final bias and amplitude values from each run folder and generates
final_fit_clean.png — a cleaner version of the final fit figure showing cluster centers as
tick marks rather than all m individual bias dots.

**Run with:** `python regenerate_figures.py`

---

### plot_convergence_now.py: Live Convergence Plot

**Purpose:** Generates a convergence plot from whatever run_meta.csv files currently exist
on disk. Safe to run at any point while simulate_parallel.py is still running.

**Output:** figures/Replication data/convergence_plot_current.png

**Run with:** `python plot_convergence_now.py`

---

## Run Order

```
python simulate_parallel.py
python verify_pruning.py
python instability_test.py
python regenerate_figures.py
python collapse_v2.py          # independent; can be run at any time
```

All scripts are safe to interrupt and restart. Already completed runs are detected via
run_meta.csv (simulations) or existing output figures (verify and instability scripts)
and skipped automatically. collapse_v2.py is fully independent of the 1D pipeline.

---

## Model and Controlled Variables

### 1D Network (Problems 4.1 and 4.3)

$$f(x) = \sum_{j=1}^{m} a_j \, \sigma(x - b_j), \quad \sigma(z) = \max(0, z), \quad x \in [-1, 1]$$

### 2D Network (Problem 4.2)

$$f(x) = \sum_{j=1}^{m} a_j \, \sigma(w_j^\top x + b_j), \quad x \in [-1, 1]^2$$

### Training

1D: gradient flow on continuous MSE loss
$$\mathcal{L} = \frac{1}{2}\int_{-1}^{1}(f(x) - f^\ast(x))^2\,dx$$

2D: Adam on empirical MSE loss over N_train=4096 uniform samples from [-1,1]²

### Gradient Flow ODEs (1D only)

$$\dot{a}_j = -\int_{-1}^{1}(f - f^\ast)\,\sigma(x - b_j)\,dx$$

$$\dot{b}_j = a_j \int_{b_j}^{1}(f - f^\ast)\,dx$$

### Variable Definitions

| Variable | What it is | Role |
|---|---|---|
| m | Network width | Primary controlled variable |
| T | Upper time limit of gradient flow (1D only) | Controls how long gradient flow runs |
| b_j | Bias parameter of neuron j (1D) | Determines kink location; collapse of b_j into clusters is the central 1D phenomenon |
| w_j, b_j | Weight vector and bias of neuron j (2D) | Together define the oriented hyperplane for that neuron |
| θⱼ, βⱼ | Canonical angle and normalized offset (2D) | Scale-invariant coordinates; collapse measured in this space |
| a_j | Amplitude of neuron j | Evolves jointly with b_j; sum of amplitudes per cluster gives effective pruned weight |
| f* | Target function | Determines k (1D) or collapse mode (2D) |
| k | Number of sign-changing zeros of (f*)'' in (-1,1) | Conjectured limit of C(m, f*) as m → ∞ (1D only) |
| C(m, f*) | Number of distinct bias clusters at end of simulation | Output being measured in 1D experiments |
| delta | Max intra-cluster diameter (1D) | Used in 4.3 pruning bound |
| T_PERTURB | Additional integration time after injection (instability_test.py only) | Controls perturbation test duration |

---

## Target Functions

### 1D Targets

| Key | Function | Inflection pts k |
|---|---|---|
| sin_1pi | sin(π x) | 1 |
| x_cubed | x³ | 1 |
| sin_2pi | sin(2π x) | 3 |
| poly_k3 | x⁵ − 3x³ | 3 |
| sin_3pi | sin(3π x) | 5 |
| sin_4pi | sin(4π x) | 7 |
| sin_5pi | sin(5π x) | 9 |
| sin_6pi | sin(6π x) | 11 |
| sin_7pi | sin(7π x) | 13 |

For sin(nπx): the second derivative has exactly 2n−1 sign-changing zeros in (−1, 1).

### 2D Targets

| Key | Function | Predicted collapse |
|---|---|---|
| ridge | sin(2πuᵀx) + 0.5sin(4πuᵀx), u=(3/5,4/5) | Directional collapse toward θ≈53° |
| separable | sin(2πx₁) + 0.3sin(2πx₂) | Two direction families, unequal sizes |
| radial | cos(π‖x‖) | No collapse; entropy near uniform |

---

## Output Files Per Run

### 1D (simulate_parallel.py)

| File | Goal | Contents |
|---|---|---|
| slide93_reproduction.png | 1 | Sorted bias trajectories, final fit vs target, MSE loss on log scale |
| clusters_vs_inflections.png | 1 | Final cluster locations vs analytically known inflection points |
| ode_verification.png | 3 | Amplitude velocity, bias velocity, and R_j at final time |
| convergence_check.csv | 3 | Per neuron b_j, a_j, da/dt, db/dt, R_j, active flag |
| run_meta.csv | All | Single row summary for restart safety |
| pruning_verification.png | 4 | Full vs pruned vs target, pointwise error, actual error vs bound |
| goal2_near.png | 2 | Bias trajectories and cluster count after near injection |
| goal2_isolated.png | 2 | Same after isolated injection |
| final_fit_clean.png | helper | Clean final fit with cluster centers as tick marks |

### 2D (collapse_v2.py)

| File | Contents |
|---|---|
| collapse_v2_results.json | Epoch-by-epoch diagnostics for all target × width combinations |
| collapse_v2_summary.png | 3×3 grid: angular entropy, cluster count, prune ratio across all targets and widths |
| collapse_v2_{target}.png | Per-target plot: loss curve, entropy trajectory, (θ,β) point cloud at final epoch, effective width panel |

---

**Note on current run status (as of 6/8/2026):**

| Script | Status |
|---|---|
| simulate_parallel.py | 🔄 69 runs complete (T=500, m up to 3500, all 9 targets); m=5000 in progress |
| simulate_discrete.py | ✅ Complete — 52 runs; C does not converge to k under discrete GD |
| verify_pruning.py | ✅ Complete — bound verified across all completed runs |
| collapse_v2.py | ✅ Complete — all 2D results analyzed |
| instability_test.py | ⏳ Pending — waiting on simulate_parallel.py to finish |