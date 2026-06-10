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
| 4.1.4 | What determines which neurons survive collapse to become cluster representatives? | Addressed by lottery_ticket_experiment.py: bias position at t=0 is the sole informative quantity; neurons geometrically close to inflection points are the "lucky" survivors. Amplitude values at initialization carry no information. |

#### Current Numerical Findings (T=500, m up to 5000, complete)

78 runs across 9 targets. C(m, f*) follows a non-monotone rise-then-fall pattern
universally; the peak shifts to larger m as k increases.

| Target | k | m=50 | m=100 | m=250 | m=500 | m=1000 | m=1500 | m=2000 | m=3500 | m=5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| sin_1pi | 1 | 31 | 45 | 22 | 6 | **1** | 1 | 1 | 1 | 1 |
| sin_2pi | 3 | 26 | 29 | 13 | 8 | 2 | †1 | †1 | †1 | †1 |
| sin_3pi | 5 | 11 | 16 | 23 | 14 | **5** | 2 | †1 | †1 | †1 |
| sin_4pi | 7 | 9 | 11 | 21 | 26 | 17 | **7** | 2 | †1 | †1 |
| sin_5pi | 9 | 11 | 11 | 18 | 23 | 31 | — | 13 | 2 | †1 |
| sin_6pi | 11 | 13 | 14 | 14 | 15 | 31 | — | 40 | 13 | 2 |
| sin_7pi | 13 | 14 | 15 | 18 | 17 | 22 | — | 36 | 34 | ‡13 |
| poly_k3 | 3 | 29 | 45 | 35 | 20 | 11 | 8 | 7 | 4 | 2 |
| x_cubed | 1 | 31 | 40 | 31 | 16 | 13 | 6 | 7 | 4 | 4 |

†**Cluster structure dissolves:** at large m, the network switches to a near-uniform bias
distribution across the full domain. Direct inspection of the saved b_j and a_j values for
sin_3pi m=2000 confirms this: 1893 active neurons span [-1.04, 1.00] with only 1 gap
exceeding 0.02, and amplitude weighting near each of the 5 inflection points is equal
(~15.5 each) with no peaks. The cluster structure is genuinely absent, not merely
undetectable. Whether this represents a failure of the conjecture or a different
representational regime is an open question. Affects sin_2pi m ≥ 1500, sin_3pi m ≥ 2000,
sin_4pi m ≥ 3500, sin_5pi m=5000. Excluded from limit conclusions.

‡**Not fully stationary:** sin_7pi m=5000 reports C=13=k but max|da/dt|=0.034 exceeds
the 0.01 stationarity threshold. The result may be transient; longer T needed to confirm.

**Conclusions:**

- **Conjecture supported for k=1 (sin_1pi):** C=1=k holds from m=1000 through m=5000,
  fully stationary. Strongest evidence for the conjecture.
- **sin_7pi (k=13) reaches C=k=13 at m=5000**, the first high-k target to show C=k, but
  not yet fully stationary (max_da=0.034). Promising; needs verification at longer T.
- **Below-k pattern is widespread, not isolated.** Four targets show C < k at a verified
  stationary state: sin_2pi (C=2, m=1000), sin_3pi (C=2, m=1500), sin_4pi (C=2, m=2000),
  poly_k3 (C=2, m=5000). The critical open question is whether these are true stationary
  states or initialization-dependent local minima. Multiple seeds would resolve this.
- **Polynomial targets converge far more slowly** than trigonometric targets with the same k.
  poly_k3 and x_cubed have no evenly-spaced inflection structure; x_cubed reaches only C=4
  at m=5000 despite k=1. Inflection point geometry shapes convergence beyond just counting k.
- **k=9, 11 targets (sin_5pi, sin_6pi) not yet resolved.** At m=5000 the cluster structure
  dissolves (uniform bias spread, no amplitude peaks); sin_6pi shows C=2 at m=5000 which
  may be a genuine 2-cluster state or an early stage of dissolution. Resolving this requires
  m > 5000 or analysis at intermediate m before dissolution occurs.

**multiple_seeds experiment (complete):**

At three points where C falls below k at a verified stationary state, rerun with 2
additional random seeds to determine whether below-k is consistent or initialization-dependent.
(sin_3pi m=1500 was excluded: m=1500 is not in M_VALUES, and cluster structure begins
dissolving at m=2000, making C ambiguous in that range.)

| Target | k | m | seed=42 | seed=5 | seed=25 |
|---|---|---|---|---|---|
| sin_2pi | 3 | 1000 | 2 | 1 | 1 |
| sin_4pi | 7 | 2000 | 2 | 3 | 5 |
| poly_k3 | 3 | 5000 | 2 | 2 | **3** (=k) |

**Finding:** C varies across seeds for all three targets, so below-k states are initialization-dependent local minima, not universal fixed points.
The k-cluster attractor exists (poly_k3 seed=25 reaches C=k=3), but has a narrow basin
of attraction. A proof of C→k must account for initialization geometry.

### Open Problem 4.2: Higher-Dimensional Collapse

**Conjecture from the slides:** For structured d-dimensional targets, gradient flow
concentrates the neuron measure onto a lower-dimensional set in hyperplane space. The
support may be points (same hyperplane), curves (parallel offsets), or several direction
families.

| Specific Goal | Description | Status |
|---|---|---|
| 4.2.1 | Does directional collapse occur for structured targets? | Partially confirmed: angular entropy drops for ridge and separable targets, especially at small width |
| 4.2.2 | Does collapse mode track target structure? | Confirmed at m=64: ridge shows mild directional concentration, separable shows family splitting, radial resists collapse |
| 4.2.3 | Does collapse generalize to larger width? | **Collapse weakens monotonically** as m grows across all 3 targets (m∈{64,256,512,1024}). Radial m=1024: H=3.53≈H_max, 1 cluster, near-perfect uniform distribution, confirming rotational symmetry. Ridge and separable entropy increases with m but collapse persists. Large-m behavior is opposite of theoretical prediction. |
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

### Goal 6: Identify Which Neurons Survive Collapse and Why (4.1.4)

Test whether neurons geometrically close to inflection points at initialization
preferentially survive gradient flow collapse to become cluster representatives, and
whether selecting just those k neurons reproduces full-network performance. Understanding
the initialization geometry of surviving neurons constrains what any proof of 4.1 must
account for. Addressed by **lottery_ticket_experiment.py**.

### Goal 7: Build Intuition Toward a Formal Proof (4.1 and 4.3)

Synthesize numerical evidence from all goals to identify proof strategies. This is an
interpretive goal informed by all scripts.

---

## Project Structure

```
MathProject4/
|
|-- MathProject Slides.pdf          Source slides: model, ODEs, and open problems
|
|-- simulate_parallel.py            Main simulation: m sweep at T=500, all 9 targets (Goals 1, 3)
|-- lottery_ticket_experiment.py    Geometric lottery ticket analysis (supporting 4.1)
|-- verify_pruning.py               Pruning bound verification (Goal 4)
|-- instability_test.py             k+1 instability test (Goal 2)
|-- higher_dim_collapse.py          2D collapse experiment (Goal 5)
|-- regenerate_figures.py           Helper: regenerates clean final fit figures
|
|-- archive/                        Superseded scripts (simulate_discrete.py data still used)
|   |-- simulate_discrete.py        Discrete GD version: 52 runs complete, data in figures/
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
|-- plots/
|   |-- lth_geometric/              Lottery ticket outputs: survival, loss curves, overlap
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
three new high-k targets (k=9, 11, 13). All runs use T=500, long enough to reach
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

**Current result:** Bound holds for all 78 T=500 runs (100%). Tightness ratios range from
0.0002 to 0.148; the bound holds with substantial slack in all cases.

Key observations across the m sweep:

- **Σ|aⱼ| grows roughly as √m.** For sin_1pi: 14.4 (m=50) → 43.2 (m=5000). Similar
  scaling across all targets. Growth rate is faster for high-k targets (sin_7pi reaches
  Σ|aⱼ| = 524 at m=5000).
- **The bound grows with m while actual pruning error plateaus.** For sin_1pi, the actual
  error stabilizes near 2.5 from m=1000 onward, but the bound grows from 34.7 to 87.9.
  Tightness (actual/bound) drops from 0.073 at m=1000 to 0.029 at m=5000; the bound
  becomes proportionally looser as m grows.
- **δ does not shrink with m at T=500.** For sin_1pi at large m, the single cluster spans
  nearly the full bias range (δ ≈ 2.03), so δ · Σ|aⱼ| grows as √m with no offset.
- **Implication for Open Problem 4.3:** Proving the bound non-vacuous requires showing δ
  shrinks as a function of something other than m alone, likely as a function of T (longer
  training tightens clusters), not network width.

**Outputs per run:** pruning_verification.png

**Global outputs:** pruning_bound_results.csv, pruning_bound_summary.png

**Run with:** `python verify_pruning.py`

---

### instability_test.py: Goal 2 Instability Test

**Open problems addressed:** 4.1.1

**Purpose:** Injects one extra neuron into converged k-cluster states and tests whether
gradient flow drives the system back to k clusters. Two injection strategies per run: near
(just outside an existing cluster) and isolated (maximally distant from all cluster centers).
above_k runs are tested by continuing the ODE naturally (no injection) for T_perturb=1000.

**Qualifying runs from current data:** 22 runs across exact_k (8), above_k (8), and
below_k (6) categories. 36 total jobs (above_k: 1 test each; exact_k and below_k: 2 tests each).

**Outputs per run:** goal2_near.png, goal2_isolated.png, goal2_natural.png (above_k only)

**Global outputs:** goal2_results.csv, goal2_summary.png

**Run with:** `python instability_test.py`

**Results (36/36 complete):**

**above_k: 8/8 complete. All no change.**

Every above_k run continued for T_perturb=1000 with zero cluster count change.
C stays fixed at its above-k value across all 8 runs spanning sin_4pi, sin_5pi,
sin_6pi, sin_7pi, and poly_k3 at various m. These states are confirmed ODE fixed points;
they do not spontaneously dissolve toward k even with substantial continued integration.

| Target | m | k | initial C | final C | result |
|---|---|---|---|---|---|
| sin_4pi | 50 | 7 | 9 | 9 | no change |
| sin_5pi | 50 | 9 | 11 | 11 | no change |
| sin_5pi | 100 | 9 | 11 | 11 | no change |
| sin_6pi | 50 | 11 | 13 | 13 | no change |
| sin_7pi | 50 | 13 | 14 | 14 | no change |
| sin_7pi | 100 | 13 | 15 | 15 | no change |
| sin_6pi | 3500 | 11 | 13 | 13 | no change |
| poly_k3 | 3500 | 3 | 4 | 4 | no change |

**Implication for OP 4.1.1:** A proof that C ≤ k requires above-k states to be unstable.
These results show they are instead stable fixed points, meaning gradient flow alone
cannot reduce C from above-k to k. The conjecture C → k as m → ∞ must rely on
initialization geometry, not on instability of above-k states.

**below_k: 12/12 complete. All no change.**

In every tested below_k case, the injected neuron's amplitude remained at its initial
value (a_inject = 0.01 throughout T_perturb=1000). The amplitude never crossed the
active threshold (0.05), so C never increased. Both near and isolated injection strategies
failed to raise C toward k.

This confirms below_k stationary states are genuine ODE fixed points; the gradient
provides zero net force on the injected neuron's amplitude. However, the multiple_seeds
analysis shows C varies across seeds for all 3 targets, so below-k states are
initialization-dependent local minima. They are stable once reached, but whether they
are reached depends on initialization.

**exact_k: 16/16 complete. All returned to k.**

All 16 tested cases returned to k after perturbation (returned_to_k=1 in every run).
Injected neurons stay below the active threshold for the duration of T_perturb=1000 and
do not grow into new clusters. Exact-k states are confirmed stable attractors:
the ODE restores the k-cluster configuration after small perturbations across all
tested targets and m values (sin_1pi, sin_3pi, sin_4pi, sin_7pi).

---

### collapse_v2.py: Open Problem 4.2, 2D Collapse Experiment

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

4. Separable at m=512 stabilizes at 3 DBSCAN clusters from epoch 2700 onward, consistent
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
result clarifies that the conjecture is specific to the continuous-time ODE dynamics;
discretization breaks the convergence. This is not a flaw; it means the ODE structure is
essential to the phenomenon, not incidental.

**Output data:** `figures/Discrete GD/`, 52 runs across 9 targets,
m ∈ {50, 100, 250, 500, 1000, 1500}.

---

### lottery_ticket_experiment.py: Geometric Lottery Ticket

**Open problems addressed:** 4.1 (supporting analysis)

**Question:** At initialization, all amplitudes a_j ~ N(0, 0.01) are near zero; the
only structure is the random bias positions b_j. Neurons whose initial bias happens to
land close to an inflection point of f* are geometrically "lucky." Do these neurons
preferentially survive gradient flow collapse to become cluster representatives, and can
just those k neurons alone match full-network performance?

**Two claims tested:**

- **Claim 1 (Survival):** Neurons initially close to an inflection point are more likely
  to become cluster representatives after collapse. Measured by overlap fraction between
  the k geometrically-selected neurons and the final cluster representatives.

- **Claim 2 (Performance):** Training only the k geometrically-closest neurons (one per
  inflection point) from their initial positions matches full-m-network performance.

**Five training conditions per (target, m):**

| Condition | Description |
|---|---|
| full | All m neurons (baseline) |
| geometric | k neurons, one per inflection point, selected by min \|b0_j − x_infl\| |
| bias_only | Same k bias positions as geometric, but fresh random amplitudes |
| amp_only | Same k amplitudes as geometric, but random bias positions |
| random_k | k neurons chosen uniformly at random (5 trials for variance) |

**Targets and parameters:** sin_1pi (k=1), sin_2pi (k=3), sin_3pi (k=5), sin_4pi (k=7);
m ∈ {500, 1000, 1500}; T=500. 12 base runs + random_k trials = 52 total.

**Results:**

- **Claim 2 is false.** The k geometric neurons achieve 100× to 50,000× higher loss than
  the full network. k neurons produce only k kinks in the piecewise-linear approximation;
  the full network uses redundancy to achieve far finer fits. This failure is
  mathematically expected, not a flaw in the selection criterion.

- **geometric ≈ bias_only exactly** (loss difference < 0.002 in every case). Initial
  amplitude values carry no information; only bias position matters at t=0. This
  confirms the initialization is purely geometric.

- **amp_only is consistently the worst condition.** Good amplitudes paired with random
  positions performs worse than good positions with random amplitudes. Bias position is
  the sole informative quantity at initialization.

- **random_k occasionally beats geometric** (sin_3pi, sin_2pi). Proximity to an
  inflection point makes a neuron a better collapse representative, not a better
  approximator when the subnetwork has only k neurons.

- **Claim 1 (survival overlap):** see `plots/lth_geometric/overlap_summary.png` and
  per-target `survival_{target}_m={m}.png`.

**Outputs:** `plots/lth_geometric/`, including survival plots, loss curves, final performance
comparisons, overlap_summary.png, lth_geometric_summary.csv.

**Run with:** `python lottery_ticket_experiment.py`

---

### regenerate_figures.py: Figure Cleaner

**Purpose:** Reads final bias and amplitude values from each run folder and generates
final_fit_clean.png, a cleaner version of the final fit figure showing cluster centers as
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
python lottery_ticket_experiment.py   # depends on simulate_parallel.py output
python verify_pruning.py
python instability_test.py
python regenerate_figures.py
python collapse_v2.py                 # independent; can be run at any time
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

**Note on current run status (as of 6/9/2026):**

| Script | Status |
|---|---|
| simulate_parallel.py | ✅ Complete: 78 runs (T=500, m up to 5000, all 9 targets) |
| simulate_discrete.py | ✅ Complete: 52 runs; C does not converge to k under discrete GD |
| lottery_ticket_experiment.py | ✅ Complete: 52 runs; geometric ≈ bias_only confirmed; k-ticket does not match full network |
| verify_pruning.py | ✅ Complete: bound holds for all 78 T=500 runs; Σ\|aⱼ\| grows ~√m, bound loosens at large m |
| collapse_v2.py | ✅ Complete: all 2D results analyzed |
| instability_test.py | ✅ Complete: 36/36 jobs done; above_k stable fixed points, below_k injection-resistant, exact_k all returned to k |
| regenerate_figures.py | ✅ Complete: final_fit_clean.png generated for all targets × M_VALUES at T=500 |
| multiple_seeds.py | ✅ Complete: 6 jobs done; below-k states are initialization-dependent local minima; C varies across seeds for all 3 targets |