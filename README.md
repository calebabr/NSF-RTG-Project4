# NSF RTG: Project 4: Bias Collapse in Shallow ReLU Networks

## Overview

This project numerically investigates two open problems from the course slides:
**Open Problem 4.1** (cluster count conjecture) and **Open Problem 4.3**
(provable pruning bound).

The setting is a 1D shallow ReLU network trained by gradient flow on a smooth
target function. The central observed phenomenon is that the bias parameters
spontaneously collapse into tight clusters over training, and the number of
clusters appears to converge toward k, the number of inflection points of the
target function, as the network width m grows. This is the subject of Open
Problem 4.1. Open Problem 4.3 then asks whether the clustered network can be
safely pruned to k neurons with a provable error bound.

---

## Coverage of the Open Problems

### Open Problem 4.1: Cluster Count

**Conjecture from the slides:**

$$\lim_{m \to \infty} C(m, f^\ast) = \lvert\{x \in [-1,1] : (f^\ast)''(x) = 0 \text{ and changes sign}\}\rvert$$

The slides list three specific goals for this open problem:

| Specific Goal | Description | Status in This Project |
|---|---|---|
| 4.1.1 | Prove C(m, f*) ≤ Cmax(f*) independent of m | Numerically supported via simulate scripts and instability_test.py; not proven |
| 4.1.2 | Is convergence to clusters finite-time or asymptotic? | Data implies asymptotic (C decreases gradually with T) but no dedicated experiment (not yet explicitly addressed) |
| 4.1.3 | Does the count depend on curvature amplitude or only the sign pattern of f''? | Not addressed yet; would require comparing targets with identical inflection locations but different curvature magnitudes |

### Open Problem 4.3: Provable Pruning

**Bound from the slides:**

$$\|\tilde{f} - f\|_{L^2} \leq \delta \cdot \sum_{j=1}^{m} |a_j|$$

| Aspect | Status in This Project |
|---|---|
| Numerically verify the bound holds | Fully addressed by verify_pruning.py: 108 of 108 runs confirmed |
| Bound the key challenge: Σ\|aⱼ\| grows with m | Tracked in summary figure; growth observed but not controlled; remains open |

---

## Research Goals and Script Mapping

Each numbered goal below corresponds to work in specific scripts.

### Goal 1: Verify the Cluster Count Conjecture (4.1 conjecture, 4.1.1)

Numerically confirm that C(m, f*) converges toward k as m grows, and that
this convergence stabilizes independently of how large m gets once past a
threshold. Addressed by **simulate.py** and **simulate_parallel.py**.

### Goal 2: Show k+1 Cluster Configurations Are Unstable (4.1.1)

Show numerically that if you inject an extra cluster into a converged
k-cluster solution, gradient flow drives the system back to k clusters rather
than maintaining the k+1 configuration. Addressed by **instability_test.py**.
This supports 4.1.1 by providing evidence that C cannot remain above k once
m is large enough.

### Goal 3: Verify Stationary Point Conditions (4.1 conjecture support)

At convergence, verify that ODE velocities are approximately zero and that the
integrated residual R_j from each bias to 1 is approximately zero for every
active neuron. This connects the numerical results to the mathematical fixed
point structure. Addressed automatically inside **simulate.py** and
**simulate_parallel.py** as part of every run.

### Goal 4: Numerically Verify the Pruning Bound (4.3)

Verify the inequality ||f_tilde minus f|| ≤ δ · Σ|aⱼ| across all
converged runs, and track how the bound components (δ and Σ|aⱼ|) behave as
m grows. Addressed by **verify_pruning.py**.

### Goal 5: Build Intuition Toward a Formal Proof (4.1 and 4.3)

Synthesize the numerical evidence from all goals to identify proof strategies
for the open problems. This is an interpretive goal informed by all scripts.

---

## Project Structure

```
MathProject4/
|
|-- MathProject Slides.pdf          Source slides: model, ODEs, and open problems
|
|-- simulate.py                     Sequential baseline simulation (Goals 1, 3)
|-- simulate_parallel.py            Parallel extended simulation (Goals 1, 3)
|-- verify_pruning.py               Pruning bound verification (Goal 4)
|-- instability_test.py             k+1 instability test (Goal 2)
|
|-- notebooks/
|   |-- 01_gradient_flow_simulator.ipynb   Early exploratory runs on x^2 target
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
|       |-- run_summary.csv                  Results from simulate.py
|       |-- run_summary_parallel.csv         Results from simulate_parallel.py
|       |-- convergence_plot.png             C(m) vs m from simulate.py
|       |-- convergence_plot_parallel.png    C(m) vs m combining both scripts
|       |-- pruning_bound_results.csv        Pruning bound metrics across all runs
|       |-- pruning_bound_summary.png        Summary plots for Open Problem 4.3
|       |-- goal2_results.csv               Instability test results across all runs
|       |-- goal2_summary.png               Summary heatmap and return rates
|
|-- data/                            Raw trajectory arrays from exploratory notebook
    |-- sol_t.npy
    |-- sol_y.npy
    |-- losses.npy
```

---

## Scripts

### simulate.py: Sequential Baseline

**Open problems addressed:** 4.1 (main conjecture), 4.1.1 (partial), Goal 3

**Purpose:** Establishes baseline numerical evidence for the conjecture across
six target functions with analytically known inflection point counts. Uses
integration times long enough to show the onset of collapse for simpler targets
and to identify which targets need longer integration.

**Targets and parameters:**

| Targets | m values | T values |
|---|---|---|
| sin(pi x) k=1, x^3 k=1 | 50, 100, 250 | 200, 500, 1000 |
| sin(2pi x) k=3, x^5 minus 3x^3 k=3 | 500, 1000, 1500 | |
| sin(3pi x) k=5, sin(4pi x) k=7 | | |

**Key finding:** sin(pi x) converges to exactly 1 cluster at m at or above
1000, T=1000. sin(4pi x) reaches exactly 7 clusters at m=1500, T=1000.
Harder targets are still mid-collapse at T=1000.

**Outputs per run:** slide93_reproduction.png, clusters_vs_inflections.png,
ode_verification.png, convergence_check.csv, run_meta.csv

**Global outputs:** run_summary.csv, convergence_plot.png

**Run with:** `python simulate.py`

---

### simulate_parallel.py: Parallel Extended Runs

**Open problems addressed:** 4.1 (main conjecture), 4.1.1 (partial), Goal 3

**Purpose:** Extends the baseline in two directions. First, it re-runs all
existing targets at T=5000 and T=10000 to determine whether the targets still
mid-collapse at T=1000 eventually converge to k. Second, it adds three new
targets with k=9, 11, and 13 to test the conjecture at higher complexity.
Multiprocessing is used because sequential execution at T=10000 with m up to
5000 would take prohibitively long.

**Why T=5000 and T=10000 only:** simulate.py already demonstrated that T at
or below 1000 does not produce convergence for harder targets. Running low T
for new targets would only confirm what is already established.

**Targets and parameters:**

| Targets | m values | T values |
|---|---|---|
| All 6 from simulate.py | 50, 100, 250, 500 | 5000, 10000 |
| sin(5pi x) k=9 | 1000, 1500, 2000, 3000, 5000 | |
| sin(6pi x) k=11 | | |
| sin(7pi x) k=13 | | |

**Outputs per run:** same four files as simulate.py, plus run_meta.csv

**Global outputs:** run_summary_parallel.csv, convergence_plot_parallel.png
(combines all T values from both scripts for the full convergence picture;
new targets show T=5000 and T=10000 only since low T was never run for them)

**Run with:** `python simulate_parallel.py`

---

### verify_pruning.py: Open Problem 4.3 Verification

**Open problems addressed:** 4.3 (pruning bound)

**Purpose:** After the simulation scripts complete, reads the saved b_j and
a_j values from each run folder and numerically verifies the pruning bound.
No re-simulation is needed. Works with outputs from both simulate.py and
simulate_parallel.py.

**What it verifies per run:**

For each run folder containing convergence_check.csv and run_meta.csv:

1. Groups neurons into clusters using the same gap tolerance as the simulations
2. Computes delta (max intra-cluster diameter) and the amplitude sum Σ|aⱼ|
3. Constructs the pruned network f_tilde (one neuron per cluster at centroid)
4. Computes the actual L2 pruning error and the bound delta times Σ|aⱼ|
5. Records whether the bound holds and the tightness ratio (actual divided by bound)

**Current result:** Bound holds for all 108 completed runs. Tightness ratios
range from 0.001 to 0.15, meaning the bound is satisfied but not tight. The
amplitude sum Σ|aⱼ| grows with m across all targets, which is the core open
challenge identified in the slides: proving the bound is non-vacuous requires
showing this growth is controlled relative to how fast delta shrinks.

**Outputs per run:** pruning_verification.png

**Global outputs:** pruning_bound_results.csv, pruning_bound_summary.png
(four panels: actual error vs bound scatter, tightness vs m, amplitude sum vs m,
cluster diameter vs m)

**Run with:** `python verify_pruning.py`

---

### instability_test.py: Goal 2 Instability Test

**Open problems addressed:** 4.1.1 (C ≤ Cmax independent of m)

**Purpose:** Operates only on runs that have fully converged to exactly k
clusters (n_clusters == k_true in run_meta.csv). For each such run, it injects
one extra neuron into the converged state to create a k+1 cluster configuration,
then continues integrating the ODE forward to see whether gradient flow drives
the system back to k clusters. This directly tests the instability claim in
4.1.1: if k+1 configurations always dissolve, that is evidence that C cannot
remain above k, which is one of the key things that would need to be proven.

**Two injection strategies per run:**

Near injection: places the extra neuron just outside the boundary of an
existing cluster (at 3 times the cluster tolerance distance). Tests whether
proximity to an existing cluster causes merging.

Isolated injection: places the extra neuron at the location in the domain
that is maximally distant from all existing cluster centers. Tests whether
the amplitude decays even when there is no nearby cluster to merge with.

**What is tracked:** bias trajectory of the injected neuron over T_PERTURB=1000
time units, amplitude of the injected neuron (does it decay toward zero?),
and the total cluster count C(t) (does it drop from k+1 back to k?).

**Qualifying runs from current data:** 10 runs across sin(pi x) k=1,
sin(3pi x) k=5, and sin(4pi x) k=7. More will qualify once
simulate_parallel.py completes.

**Outputs per run:** goal2_near.png, goal2_isolated.png

**Global outputs:** goal2_results.csv, goal2_summary.png (heatmap of which
runs returned to k, and bar chart of return rate by injection type)

**Run order:** run after both simulate scripts and verify_pruning.py

**Run with:** `python instability_test.py`

---

## Model

**Network:**

$$f(x) = \sum_{j=1}^{m} a_j \, \sigma(x - b_j), \quad \sigma(z) = \max(0, z), \quad x \in [-1, 1]$$

**Training via gradient flow on continuous MSE loss:**

$$\mathcal{L} = \frac{1}{2}\int_{-1}^{1}(f(x) - f^\ast(x))^2\,dx$$

**Gradient flow ODEs:**

$$\dot{a}_j = -\int_{-1}^{1}(f - f^\ast)\,\sigma(x - b_j)\,dx \qquad \dot{b}_j = a_j \int_{b_j}^{1}(f - f^\ast)\,dx$$

---

## Target Functions

| Key | Function | Inflection pts k | Introduced in |
|---|---|---|---|
| sin_1pi | sin(pi x) | 1 | simulate.py |
| x_cubed | x^3 | 1 | simulate.py |
| sin_2pi | sin(2 pi x) | 3 | simulate.py |
| poly_k3 | x^5 minus 3x^3 | 3 | simulate.py |
| sin_3pi | sin(3 pi x) | 5 | simulate.py |
| sin_4pi | sin(4 pi x) | 7 | simulate.py |
| sin_5pi | sin(5 pi x) | 9 | simulate_parallel.py |
| sin_6pi | sin(6 pi x) | 11 | simulate_parallel.py |
| sin_7pi | sin(7 pi x) | 13 | simulate_parallel.py |

For sin(n pi x): the second derivative is negative n squared pi squared times
sin(n pi x), which has exactly 2n minus 1 sign-changing zeros in the open
interval from negative 1 to 1.

---

## Output Files Per Run

| File | Goal | Contents |
|---|---|---|
| slide93_reproduction.png | 1 | Sorted bias trajectories, final fit vs target, MSE loss on log scale |
| clusters_vs_inflections.png | 1 | Final cluster locations vs analytically known inflection points |
| ode_verification.png | 3 | Amplitude velocity, bias velocity, and integrated residual R_j at final time |
| convergence_check.csv | 3 | Per neuron b_j, a_j, da/dt, db/dt, R_j, active flag, R near zero flag |
| run_meta.csv | All | Single row summary enabling Ctrl+C safe restart without repeating completed runs |
| pruning_verification.png | 4 | Full vs pruned vs target, pointwise error, actual error vs bound bar chart |
| goal2_near.png | 2 | Bias trajectories, injected amplitude, and cluster count after near injection |
| goal2_isolated.png | 2 | Same panels after isolated injection |

---

## Run Order

```
python simulate.py
python simulate_parallel.py
python verify_pruning.py
python instability_test.py
```

Each script is safe to interrupt and restart. Already completed runs are
detected via run_meta.csv (simulations) or existing output figures (verify
and instability scripts) and skipped automatically.
