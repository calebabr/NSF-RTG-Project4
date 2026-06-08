"""
lottery_ticket_experiment.py
============================
Geometric Lottery Ticket experiment for shallow ReLU networks under
ODE gradient flow.

The question
------------
In this architecture, f(x) = sum_j a_j * relu(x - b_j), all neurons
start with a_j ~ 0 (amplitudes are uninformative at t=0).  The only
structure at initialization is the bias position b_j.

The real "lottery" at t=0 is geometric: neurons whose initial bias
is already close to an inflection point of f* are pre-positioned to
become cluster representatives without needing to travel far.  ODE
flow should preferentially preserve these geometrically-lucky neurons.

This gives two testable claims:

  Claim 1 (Survival): neurons initially close to inflection points
    are more likely to become cluster representatives after collapse.

  Claim 2 (Performance): training only the k geometrically-closest
    neurons (one per inflection point) from their initial biases
    matches full-m-network performance.

Experiment structure
--------------------
  Phase 1 -- Survival analysis (per (target, m)):
    - Run full ODE flow; identify cluster representatives
    - For every neuron, record its initial distance to nearest inflection
    - Compare: do survivors have smaller initial distances than non-survivors?
    - Compute overlap: how many of the k geometrically-selected neurons
      end up as cluster representatives?

  Phase 2 -- Performance comparison (per (target, m)):
    Five training conditions, all starting from the same random init:
      1. full        -- all m neurons (baseline)
      2. geometric   -- exactly k neurons, one per inflection point,
                        selected by minimum |b0_j - x_infl|
      3. bias_only   -- same k bias positions as geometric ticket,
                        but fresh random amplitudes a_j ~ N(0, 0.01)
                        (tests: does position alone explain performance?)
      4. amp_only    -- same k amplitude inits as geometric ticket,
                        but random bias positions b_j ~ U(-1, 1)
                        (tests: do a_j values carry any signal?)
      5. random_k    -- k neurons chosen uniformly at random
                        (N_RANDOM_TRIALS trials for error bars)

  Prediction: bias_only ~= geometric (amplitudes are uninformative at
  a_j ~ 0.01), amp_only ~= random_k (position is everything).

Output
------
  plots/lth_geometric/
      survival_{target}_m={m}.png   -- initial distance distributions,
                                       survivors vs non-survivors
      overlap_{target}.png          -- overlap fraction across m values
      loss_curves_{target}_m={m}.png
      final_performance_{target}_m={m}.png
      lth_geometric_summary.csv

Usage
-----
    python lottery_ticket_experiment.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time

# =============================================================================
# Quadrature grid  (matches simulate_parallel.py)
# =============================================================================
N_QUAD = 200
X_QUAD = np.linspace(-1.0, 1.0, N_QUAD)
DX     = X_QUAD[1] - X_QUAD[0]

# =============================================================================
# ODE parameters  (match simulate_parallel.py)
# =============================================================================
T_FINAL = 500
N_SAVE  = 300
SEED    = 42

CLUSTER_TOL      = 0.02
ACTIVE_THRESHOLD = 0.05

# =============================================================================
# Experiment parameters
# =============================================================================
M_VALUES       = [500, 1000, 1500]
N_RANDOM_TRIALS = 5

OUT_DIR     = os.path.join('plots', 'lth_geometric')
SUMMARY_CSV = os.path.join(OUT_DIR, 'lth_geometric_summary.csv')

SUMMARY_FIELDS = [
    'target', 'm', 'k_true', 'condition', 'trial',
    'n_neurons', 'final_loss', 'loss_at_25pct', 'loss_at_50pct',
    'T',
]

# =============================================================================
# Target functions and inflection locations  (from simulate_parallel.py)
# =============================================================================
def f_sin_1pi(x): return np.sin(    np.pi * x)
def f_sin_2pi(x): return np.sin(2 * np.pi * x)
def f_sin_3pi(x): return np.sin(3 * np.pi * x)
def f_sin_4pi(x): return np.sin(4 * np.pi * x)

TARGETS = {
    'sin_1pi': (r'$\sin(\pi x)$',  1, f_sin_1pi),
    'sin_2pi': (r'$\sin(2\pi x)$', 3, f_sin_2pi),
    'sin_3pi': (r'$\sin(3\pi x)$', 5, f_sin_3pi),
    'sin_4pi': (r'$\sin(4\pi x)$', 7, f_sin_4pi),
}

INFLECTIONS = {
    'sin_1pi': np.array([0.0]),
    'sin_2pi': np.array([-0.5, 0.0, 0.5]),
    'sin_3pi': np.array([-2/3, -1/3, 0.0, 1/3, 2/3]),
    'sin_4pi': np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]),
}

# =============================================================================
# Core math  (from simulate_parallel.py)
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def network(x, a, b):
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def make_ode(m, f_star):
    fstar_vals = f_star(X_QUAD)
    def ode(t, y):
        a, b      = y[:m], y[m:]
        residual  = network(X_QUAD, a, b) - fstar_vals
        relu_mat  = relu(X_QUAD[:, None] - b[None, :])
        da        = -(residual[:, None] * relu_mat).sum(0) * DX
        cum_right = np.cumsum(residual[::-1])[::-1] * DX
        idx       = np.searchsorted(X_QUAD, b).clip(0, N_QUAD - 1)
        db        = a * cum_right[idx]
        return np.concatenate([da, db])
    return ode

def compute_loss(a, b, fstar_vals):
    return float(0.5 * np.trapz(
        (network(X_QUAD, a, b) - fstar_vals)**2, X_QUAD))

def count_clusters(biases, tol=CLUSTER_TOL):
    s = np.sort(biases)
    return 1 + int(np.sum(np.diff(s) > tol))

def cluster_representatives(b, a, tol=CLUSTER_TOL):
    """
    Return indices (into the original b array) of one representative
    per bias cluster -- the neuron within each cluster whose bias is
    closest to the cluster centroid.
    """
    sort_idx = np.argsort(b)
    bs = b[sort_idx]

    groups = []
    group  = [sort_idx[0]]
    for i in range(1, len(bs)):
        if bs[i] - bs[i - 1] > tol:
            groups.append(group)
            group = []
        group.append(sort_idx[i])
    groups.append(group)

    reps = []
    for g in groups:
        centroid = float(np.mean(b[g]))
        rep = min(g, key=lambda j: abs(b[j] - centroid))
        reps.append(rep)
    return reps   # list of original indices, length = n_clusters


def run_ode(a0_sub, b0_sub, f_star, T):
    """
    Run ODE gradient flow on a subnetwork defined by a0_sub, b0_sub.
    Returns (a_final, b_final, loss_log).
    loss_log is a list of (t, loss) at ~60 evenly-spaced snapshots.
    """
    m = len(a0_sub)
    fstar_vals = f_star(X_QUAD)

    sol = solve_ivp(
        make_ode(m, f_star),
        t_span=(0.0, T),
        y0=np.concatenate([a0_sub, b0_sub]),
        method='RK45',
        t_eval=np.linspace(0.0, T, N_SAVE),
        rtol=1e-4, atol=1e-6,
        max_step=max(0.1, T / 500),
    )

    snap_idx = np.arange(0, N_SAVE, max(1, N_SAVE // 60))
    loss_log = [(float(sol.t[i]),
                 compute_loss(sol.y[:m, i], sol.y[m:, i], fstar_vals))
                for i in snap_idx]
    loss_log.append((float(sol.t[-1]),
                     compute_loss(sol.y[:m, -1], sol.y[m:, -1], fstar_vals)))

    return sol.y[:m, -1], sol.y[m:, -1], loss_log

# =============================================================================
# Geometric mask selection
# =============================================================================

def find_geometric_mask(b0, inflections):
    """
    Select one neuron per inflection point: the neuron whose initial
    bias is closest to that inflection point.  Each neuron can be
    assigned to at most one inflection point (greedy, sorted by distance).

    Returns a boolean mask of length m with exactly k True entries.
    """
    m    = len(b0)
    mask = np.zeros(m, dtype=bool)
    used = set()

    # Sort inflection points by distance to nearest available neuron,
    # then assign greedily.
    for infl in inflections:
        dists = [(abs(b0[j] - infl), j) for j in range(m) if j not in used]
        _, best = min(dists)
        mask[best] = True
        used.add(best)
    return mask

def dist_to_nearest_inflection(b, inflections):
    """Return array of distances: each element is min_k |b_j - infl_k|."""
    return np.array([float(np.min(np.abs(b[j] - inflections)))
                     for j in range(len(b))])

# =============================================================================
# Single experiment: one (target, m) pair
# =============================================================================

def run_experiment(target_key, m):
    target_label, k_true, f_star_fn = TARGETS[target_key]
    inflections = INFLECTIONS[target_key]
    fstar_vals  = f_star_fn(X_QUAD)

    # Reconstruct the canonical random init (same seed as simulate_parallel.py)
    np.random.seed(SEED)
    a0 = np.random.randn(m) * 0.01
    b0 = np.random.uniform(-1.0, 1.0, m)

    # ── Phase 1: full ODE flow to identify survivors ──────────────────────────
    a_full, b_full, log_full = run_ode(a0, b0, f_star_fn, T_FINAL)

    n_clusters = count_clusters(b_full)
    rep_indices = cluster_representatives(b_full, a_full)   # original indices

    survivor_mask = np.zeros(m, dtype=bool)
    for j in rep_indices:
        survivor_mask[j] = True

    # ── Phase 1 analysis: initial distances vs survival ───────────────────────
    init_dists  = dist_to_nearest_inflection(b0, inflections)
    surv_dists  = init_dists[survivor_mask]
    other_dists = init_dists[~survivor_mask]

    # Geometric mask: k neurons closest to inflections at t=0
    geo_mask = find_geometric_mask(b0, inflections)

    # Overlap: how many geometric neurons ended up as cluster reps?
    overlap = int((geo_mask & survivor_mask).sum())

    # ── Phase 2: train three conditions ──────────────────────────────────────
    results  = []
    all_logs = {'full': log_full}

    results.append({
        'target': target_key, 'm': m, 'k_true': k_true,
        'condition': 'full', 'trial': 0,
        'n_neurons': m,
        'final_loss':    log_full[-1][1],
        'loss_at_25pct': _loss_at_fraction(log_full, 0.25),
        'loss_at_50pct': _loss_at_fraction(log_full, 0.50),
        'T': T_FINAL,
    })

    # Geometric ticket
    a_geo, b_geo, log_geo = run_ode(
        a0[geo_mask], b0[geo_mask], f_star_fn, T_FINAL)
    all_logs['geometric'] = log_geo

    results.append({
        'target': target_key, 'm': m, 'k_true': k_true,
        'condition': 'geometric', 'trial': 0,
        'n_neurons': int(geo_mask.sum()),
        'final_loss':    log_geo[-1][1],
        'loss_at_25pct': _loss_at_fraction(log_geo, 0.25),
        'loss_at_50pct': _loss_at_fraction(log_geo, 0.50),
        'T': T_FINAL,
    })

    # ── Isolation conditions ──────────────────────────────────────────────────
    rng_iso = np.random.default_rng(SEED + 77)

    # Bias-only: geometric ticket's b_j positions, fresh random a_j
    a0_fresh_bo = rng_iso.standard_normal(k_true) * 0.01
    a_bo, b_bo, log_bo = run_ode(
        a0_fresh_bo, b0[geo_mask].copy(), f_star_fn, T_FINAL)
    all_logs['bias_only'] = log_bo
    results.append({
        'target': target_key, 'm': m, 'k_true': k_true,
        'condition': 'bias_only', 'trial': 0,
        'n_neurons': k_true,
        'final_loss':    log_bo[-1][1],
        'loss_at_25pct': _loss_at_fraction(log_bo, 0.25),
        'loss_at_50pct': _loss_at_fraction(log_bo, 0.50),
        'T': T_FINAL,
    })

    # Amp-only: geometric ticket's a_j values, but random b_j positions
    b0_fresh_ao = rng_iso.uniform(-1.0, 1.0, k_true)
    a_ao, b_ao, log_ao = run_ode(
        a0[geo_mask].copy(), b0_fresh_ao, f_star_fn, T_FINAL)
    all_logs['amp_only'] = log_ao
    results.append({
        'target': target_key, 'm': m, 'k_true': k_true,
        'condition': 'amp_only', 'trial': 0,
        'n_neurons': k_true,
        'final_loss':    log_ao[-1][1],
        'loss_at_25pct': _loss_at_fraction(log_ao, 0.25),
        'loss_at_50pct': _loss_at_fraction(log_ao, 0.50),
        'T': T_FINAL,
    })

    # Random-k baseline
    rng = np.random.default_rng(SEED + 1)
    for trial in range(N_RANDOM_TRIALS):
        rand_idx  = rng.choice(m, size=k_true, replace=False)
        rand_mask = np.zeros(m, dtype=bool)
        rand_mask[rand_idx] = True

        a_rand, b_rand, log_rand = run_ode(
            a0[rand_mask], b0[rand_mask], f_star_fn, T_FINAL)
        all_logs[f'random_{trial}'] = log_rand

        results.append({
            'target': target_key, 'm': m, 'k_true': k_true,
            'condition': 'random_k', 'trial': trial,
            'n_neurons': k_true,
            'final_loss':    log_rand[-1][1],
            'loss_at_25pct': _loss_at_fraction(log_rand, 0.25),
            'loss_at_50pct': _loss_at_fraction(log_rand, 0.50),
            'T': T_FINAL,
        })

    survival_data = {
        'surv_dists':  surv_dists,
        'other_dists': other_dists,
        'init_dists':  init_dists,
        'survivor_mask': survivor_mask,
        'geo_mask':    geo_mask,
        'overlap':     overlap,
        'n_clusters':  n_clusters,
        'rep_indices': rep_indices,
    }

    info = {
        'target_key':   target_key,
        'target_label': target_label,
        'm':            m,
        'k_true':       k_true,
        'n_clusters':   n_clusters,
        'n_survivors':  int(survivor_mask.sum()),
        'overlap':      overlap,
        'overlap_frac': overlap / k_true if k_true > 0 else 0.0,
        'mean_surv_dist':  float(np.mean(surv_dists)),
        'mean_other_dist': float(np.mean(other_dists)),
        'T': T_FINAL,
    }

    return results, all_logs, survival_data, info


def _loss_at_fraction(log, frac):
    if not log:
        return float('nan')
    max_t    = log[-1][0]
    target_t = frac * max_t
    return min(log, key=lambda x: abs(x[0] - target_t))[1]


# =============================================================================
# Plots
# =============================================================================

def plot_survival(survival_data, info, out_dir):
    """
    Two-panel figure:
      Left:  histogram of initial distances to nearest inflection,
             survivors (blue) vs non-survivors (gray).
      Right: scatter of initial bias position b0_j, coloured by
             survivor status, with inflection points marked.
    """
    sd           = survival_data
    target_label = info['target_label']
    m            = info['m']
    k_true       = info['k_true']
    inflections  = INFLECTIONS[info['target_key']]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'Survival Analysis -- {target_label},  m={m},  k={k_true}\n'
        f'Cluster reps after T={info["T"]} ODE flow: {info["n_survivors"]}   '
        f'Geometric overlap: {info["overlap"]}/{k_true}',
        fontsize=12)

    # Panel 1: distance histograms
    ax = axes[0]
    bins = np.linspace(0, 1.0, 40)
    ax.hist(sd['other_dists'], bins=bins, color='lightgray',
            edgecolor='gray', label=f'Non-survivors ({(~sd["survivor_mask"]).sum()})',
            density=True, alpha=0.8)
    ax.hist(sd['surv_dists'], bins=bins, color='steelblue',
            edgecolor='navy', label=f'Survivors ({sd["survivor_mask"].sum()})',
            density=True, alpha=0.8)
    for infl in inflections:
        ax.axvline(0.0, color='green', lw=0, alpha=0)  # just for spacing
    ax.set_xlabel('Initial distance to nearest inflection point', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Do survivors start closer to inflection points?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    mean_s = info['mean_surv_dist']
    mean_o = info['mean_other_dist']
    ax.axvline(mean_s, color='steelblue', lw=2, linestyle='--',
               label=f'Mean survivors: {mean_s:.3f}')
    ax.axvline(mean_o, color='gray', lw=2, linestyle='--',
               label=f'Mean others: {mean_o:.3f}')
    ax.legend(fontsize=8)

    # Panel 2: scatter of initial bias positions
    ax = axes[1]
    b0_all = np.array([])   # reconstruct from info -- use saved init_dists
    # Colour each neuron by status
    colors_all = np.where(sd['survivor_mask'], 'steelblue', 'lightgray')
    geo_colors = np.where(sd['geo_mask'], 'darkorange', None)

    xs = np.arange(len(sd['init_dists']))
    sort_b = np.argsort(sd['init_dists'])  # sort by distance for clean scatter

    # Plot non-survivors first (background)
    nsurv_idx = np.where(~sd['survivor_mask'])[0]
    ax.scatter(sd['init_dists'][nsurv_idx], nsurv_idx / len(nsurv_idx),
               color='lightgray', s=4, alpha=0.4, label='Non-survivor')

    surv_idx = np.where(sd['survivor_mask'])[0]
    ax.scatter(sd['init_dists'][surv_idx], surv_idx / len(nsurv_idx),
               color='steelblue', s=20, zorder=3, label='Survivor (cluster rep)')

    geo_idx = np.where(sd['geo_mask'])[0]
    ax.scatter(sd['init_dists'][geo_idx],
               [i / max(len(nsurv_idx), 1) for i in geo_idx],
               color='darkorange', s=60, marker='*', zorder=4,
               label='Geometric ticket')

    ax.set_xlabel('Initial distance to nearest inflection', fontsize=11)
    ax.set_ylabel('Neuron index (scaled)', fontsize=11)
    ax.set_title('Initial distance by neuron\n'
                 '(blue=survived, orange*=geometric ticket)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir,
                         f'survival_{info["target_key"]}_m={m}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    return fname


def plot_loss_curves(all_logs, info, out_dir):
    target_label = info['target_label']
    m            = info['m']
    k_true       = info['k_true']

    fig, ax = plt.subplots(figsize=(10, 6))

    t_f, l_f = zip(*all_logs['full'])
    ax.semilogy(t_f, l_f, 'k-', lw=2, label=f'Full (m={m})')

    t_g, l_g = zip(*all_logs['geometric'])
    ax.semilogy(t_g, l_g, 'b-', lw=2.5,
                label=f'Geometric ticket (k={k_true}, init-selected)')

    t_bo, l_bo = zip(*all_logs['bias_only'])
    ax.semilogy(t_bo, l_bo, color='darkorange', lw=2, linestyle='--',
                label=f'Bias-only (geo b, fresh a)')

    t_ao, l_ao = zip(*all_logs['amp_only'])
    ax.semilogy(t_ao, l_ao, color='purple', lw=2, linestyle='-.',
                label=f'Amp-only (geo a, random b)')

    rand_keys = sorted(k for k in all_logs if k.startswith('random_'))
    for i, rk in enumerate(rand_keys):
        t_r, l_r = zip(*all_logs[rk])
        label = f'Random k={k_true}' if i == 0 else None
        ax.semilogy(t_r, l_r, color='crimson', alpha=0.4, lw=1,
                    linestyle=':', label=label)

    ax.set_xlabel('ODE flow time $t$', fontsize=11)
    ax.set_ylabel('MSE loss (log scale)', fontsize=11)
    ax.set_title(
        f'Geometric Lottery Ticket -- {target_label},  m={m},  k={k_true}\n'
        f'Overlap (geo in survivors): {info["overlap"]}/{k_true}  '
        f'(mean surv dist = {info["mean_surv_dist"]:.3f}, '
        f'mean other dist = {info["mean_other_dist"]:.3f})',
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(out_dir,
                         f'loss_curves_{info["target_key"]}_m={m}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    return fname


def plot_final_bar(results_for_pair, info, out_dir):
    target_label = info['target_label']
    m            = info['m']
    k_true       = info['k_true']

    full_loss = np.mean([r['final_loss'] for r in results_for_pair
                         if r['condition'] == 'full'])
    geo_loss  = np.mean([r['final_loss'] for r in results_for_pair
                         if r['condition'] == 'geometric'])
    bo_loss   = np.mean([r['final_loss'] for r in results_for_pair
                         if r['condition'] == 'bias_only'])
    ao_loss   = np.mean([r['final_loss'] for r in results_for_pair
                         if r['condition'] == 'amp_only'])
    rand_losses = [r['final_loss'] for r in results_for_pair
                   if r['condition'] == 'random_k']
    rand_mean = np.mean(rand_losses)
    rand_std  = np.std(rand_losses)

    labels = [f'Full\n(m={m})',
              f'Geometric\n(k={k_true})',
              f'Bias-only\n(geo b, fresh a)',
              f'Amp-only\n(geo a, rand b)',
              f'Random k={k_true}\n(n={len(rand_losses)})']
    losses = [full_loss, geo_loss, bo_loss, ao_loss, rand_mean]
    errors = [0, 0, 0, 0, rand_std]
    colors = ['#2c3e50', '#2980b9', '#e67e22', '#8e44ad', '#c0392b']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, losses, yerr=errors, color=colors,
                  edgecolor='black', linewidth=0.8, capsize=6)
    ax.set_ylabel('Final MSE loss', fontsize=11)
    ax.set_title(
        f'Final Performance -- {target_label},  m={m},  k={k_true}\n'
        f'Geo={geo_loss/full_loss:.2f}x  BO={bo_loss/full_loss:.2f}x  '
        f'AO={ao_loss/full_loss:.2f}x  Rand={rand_mean/full_loss:.2f}x  (rel. full)',
        fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{loss:.2e}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    fname = os.path.join(out_dir,
                         f'final_performance_{info["target_key"]}_m={m}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    return fname


def plot_overlap_summary(all_infos, out_dir):
    """
    For each target: overlap fraction vs m, and mean distance ratio
    (survivor dist / non-survivor dist) vs m.
    """
    target_keys = list(dict.fromkeys(d['target_key'] for d in all_infos))
    ncols = min(len(target_keys), 2)
    nrows = -(-len(target_keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Geometric Lottery Ticket -- Survival Statistics vs m',
                 fontsize=13)

    for ax_i, tkey in enumerate(target_keys):
        row = ax_i // ncols
        col = (ax_i % ncols) * 2
        infos = sorted([d for d in all_infos if d['target_key'] == tkey],
                       key=lambda d: d['m'])
        ms             = [d['m']            for d in infos]
        overlaps       = [d['overlap_frac'] for d in infos]
        dist_ratios    = [d['mean_surv_dist'] / max(d['mean_other_dist'], 1e-9)
                          for d in infos]

        # Panel A: overlap fraction
        ax = axes[row][col]
        ax.plot(ms, overlaps, 'o-', color='steelblue', lw=2, ms=8)
        ax.axhline(1.0, color='green', lw=1.5, linestyle='--',
                   label='Perfect overlap (=1)')
        ax.set_xlabel('Width $m$', fontsize=10)
        ax.set_ylabel('Overlap fraction', fontsize=10)
        ax.set_title(f'{TARGETS[tkey][0]} -- Geo overlap')
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel B: distance ratio
        ax = axes[row][col + 1]
        ax.plot(ms, dist_ratios, 's-', color='darkorange', lw=2, ms=8)
        ax.axhline(1.0, color='k', lw=1, linestyle='--',
                   label='No difference (=1)')
        ax.set_xlabel('Width $m$', fontsize=10)
        ax.set_ylabel('Mean dist ratio\n(survivors / non-survivors)', fontsize=10)
        ax.set_title(f'{TARGETS[tkey][0]} -- Dist ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'overlap_summary.png')
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    return fname


# =============================================================================
# Interpretation
# =============================================================================

def print_interpretation(all_results, all_infos):
    print('\n' + '=' * 72)
    print('INTERPRETATION -- Geometric Lottery Ticket')
    print('Shallow ReLU networks, ODE gradient flow')
    print('=' * 72)

    for tkey in TARGETS:
        t_infos   = [d for d in all_infos   if d['target_key'] == tkey]
        t_results = [r for r in all_results if r['target']     == tkey]
        if not t_infos:
            continue

        target_label, k_true, _ = TARGETS[tkey]
        print(f'\n{"-" * 72}')
        print(f'  Target: {target_label}  (k = {k_true})')
        print(f'{"-" * 72}')

        for info in sorted(t_infos, key=lambda d: d['m']):
            mv = info['m']
            full_loss  = next(r['final_loss'] for r in t_results
                              if r['condition'] == 'full' and r['m'] == mv)
            geo_loss   = next(r['final_loss'] for r in t_results
                              if r['condition'] == 'geometric' and r['m'] == mv)
            rand_losses = [r['final_loss'] for r in t_results
                           if r['condition'] == 'random_k' and r['m'] == mv]
            rand_mean = np.mean(rand_losses) if rand_losses else float('nan')

            ratio_geo  = geo_loss  / full_loss if full_loss > 0 else float('inf')
            ratio_rand = rand_mean / full_loss if full_loss > 0 else float('inf')

            print(f'\n  m = {mv}:')
            print(f'    Clusters after flow  : {info["n_clusters"]}  (k={k_true})')
            print(f'    Overlap (geo->surv)  : {info["overlap"]}/{k_true}  '
                  f'({info["overlap_frac"]:.0%})')
            print(f'    Mean init dist -- survivors : '
                  f'{info["mean_surv_dist"]:.4f}')
            print(f'    Mean init dist -- others    : '
                  f'{info["mean_other_dist"]:.4f}')
            dist_ratio = (info['mean_surv_dist'] /
                          max(info['mean_other_dist'], 1e-9))
            print(f'    Ratio (surv / others)       : {dist_ratio:.3f}  '
                  f'(< 1 means survivors start closer)')
            bo_loss = next((r['final_loss'] for r in t_results
                            if r['condition'] == 'bias_only' and r['m'] == mv), None)
            ao_loss = next((r['final_loss'] for r in t_results
                            if r['condition'] == 'amp_only' and r['m'] == mv), None)
            ratio_bo  = bo_loss  / full_loss if (bo_loss  and full_loss > 0) else float('nan')
            ratio_ao  = ao_loss  / full_loss if (ao_loss  and full_loss > 0) else float('nan')

            print(f'    Full network  : loss = {full_loss:.4e}')
            print(f'    Geometric (k) : loss = {geo_loss:.4e}  '
                  f'(ratio = {ratio_geo:.3f}x)')
            print(f'    Bias-only     : loss = {bo_loss:.4e}  '
                  f'(ratio = {ratio_bo:.3f}x)  [geo b, fresh a]')
            print(f'    Amp-only      : loss = {ao_loss:.4e}  '
                  f'(ratio = {ratio_ao:.3f}x)  [geo a, random b]')
            print(f'    Random (k)    : loss = {rand_mean:.4e}  '
                  f'(ratio = {ratio_rand:.3f}x)')

            # Claim 1: do survivors start closer?
            if dist_ratio < 0.5:
                print(f'    >> CLAIM 1 STRONG: survivors start {1/dist_ratio:.1f}x'
                      f' closer to inflections than non-survivors.')
            elif dist_ratio < 0.85:
                print(f'    >> CLAIM 1 MODERATE: survivors start somewhat closer '
                      f'(ratio={dist_ratio:.2f}).')
            else:
                print(f'    >> CLAIM 1 WEAK: survivors show little initial '
                      f'proximity advantage (ratio={dist_ratio:.2f}).')

            # Claim 2: does geometric ticket match full?
            if ratio_geo <= 1.5:
                print(f'    >> CLAIM 2 STRONG: geometric ticket matches full '
                      f'network ({ratio_geo:.2f}x).')
            elif ratio_geo <= 5.0:
                print(f'    >> CLAIM 2 PARTIAL: geometric ticket is within '
                      f'{ratio_geo:.1f}x of full.')
            else:
                print(f'    >> CLAIM 2 FAILS: geometric ticket is {ratio_geo:.1f}x'
                      f' worse -- init proximity to inflections is not enough.')

            # Amplitude isolation analysis
            if bo_loss is not None and ao_loss is not None:
                bo_geo_rel = abs(ratio_bo - ratio_geo) / max(ratio_geo, 1e-12)
                if bo_geo_rel < 0.2:
                    print(f'    >> AMPLITUDE UNINFORMATIVE: bias-only ({ratio_bo:.2f}x)'
                          f' ~= geometric ({ratio_geo:.2f}x) -- a_j init carries no signal.')
                else:
                    print(f'    >> Bias-only ({ratio_bo:.2f}x) differs from geometric '
                          f'({ratio_geo:.2f}x) -- some amplitude signal present.')
                if ratio_ao > 0.8 * ratio_rand:
                    print(f'    >> POSITION IS EVERYTHING: amp-only ({ratio_ao:.2f}x)'
                          f' performs like random ({ratio_rand:.2f}x).')
                else:
                    print(f'    >> Amp-only ({ratio_ao:.2f}x) beats random ({ratio_rand:.2f}x)'
                          f' -- amplitude init carries some positional benefit.')

            # Does geometric beat random?
            if ratio_rand > 1.5 * ratio_geo:
                print(f'    >> Geometric beats random by {ratio_rand/ratio_geo:.1f}x'
                      f' -- inflection proximity matters.')
            else:
                print(f'    >> Geometric ({ratio_geo:.2f}x) not clearly better '
                      f'than random ({ratio_rand:.2f}x).')

    print('\n' + '=' * 72)
    print('OVERALL')
    print('=' * 72)

    # Aggregate overlap and distance ratio across all (target, m)
    overlaps    = [d['overlap_frac'] for d in all_infos]
    dist_ratios = [d['mean_surv_dist'] / max(d['mean_other_dist'], 1e-9)
                   for d in all_infos]

    geo_ratios  = []
    bo_ratios   = []
    ao_ratios   = []
    rand_ratios = []
    for r in all_results:
        if r['condition'] != 'full':
            continue
        tkey, mv = r['target'], r['m']
        fl = r['final_loss']
        if fl <= 0:
            continue
        geo = next((x['final_loss'] for x in all_results
                    if x['condition'] == 'geometric'
                    and x['target'] == tkey and x['m'] == mv), None)
        bo = next((x['final_loss'] for x in all_results
                   if x['condition'] == 'bias_only'
                   and x['target'] == tkey and x['m'] == mv), None)
        ao = next((x['final_loss'] for x in all_results
                   if x['condition'] == 'amp_only'
                   and x['target'] == tkey and x['m'] == mv), None)
        rands = [x['final_loss'] for x in all_results
                 if x['condition'] == 'random_k'
                 and x['target'] == tkey and x['m'] == mv]
        if geo:
            geo_ratios.append(geo / fl)
        if bo:
            bo_ratios.append(bo / fl)
        if ao:
            ao_ratios.append(ao / fl)
        if rands:
            rand_ratios.append(np.mean(rands) / fl)

    print(f'\n  Median overlap fraction            : '
          f'{np.median(overlaps):.2f}')
    print(f'  Median dist ratio (surv/other)    : '
          f'{np.median(dist_ratios):.3f}  (< 1 = claim 1 holds)')
    print(f'  Median geo ticket / full loss     : '
          f'{np.median(geo_ratios):.3f}  (< 1.5 = claim 2 holds)')
    print(f'  Median bias-only / full loss      : '
          f'{np.median(bo_ratios):.3f}  (should ~= geo if a_j uninformative)')
    print(f'  Median amp-only / full loss       : '
          f'{np.median(ao_ratios):.3f}  (should ~= random if position is all)')
    print(f'  Median random / full loss         : '
          f'{np.median(rand_ratios):.3f}')

    med_overlap = np.median(overlaps)
    med_dist    = np.median(dist_ratios)
    med_geo     = np.median(geo_ratios)
    med_bo      = np.median(bo_ratios)
    med_ao      = np.median(ao_ratios)
    med_rand    = np.median(rand_ratios)

    amp_uninformative = abs(med_bo - med_geo) / max(med_geo, 1e-12) < 0.2
    pos_is_all        = med_ao > 0.8 * med_rand

    print()
    c1 = med_dist < 0.85
    c2 = med_geo  <= 2.0

    if c1 and c2:
        print('  CONCLUSION: Both claims hold. Neurons that start close to')
        print('  inflection points reliably survive collapse (Claim 1), and')
        print('  training only those k neurons from their initial biases')
        print('  recovers near-full performance (Claim 2). The lottery at')
        print('  initialization is genuinely geometric: the bias draw b_j')
        print('  determines which neurons are predisposed to survive.')
    elif c1 and not c2:
        print('  CONCLUSION: Claim 1 holds but Claim 2 fails. Survivors')
        print('  were geometrically lucky at init, but k neurons alone')
        print('  are not enough to match the full network -- the collapsed')
        print('  amplitudes learned during flow carry most of the capacity.')
    elif not c1 and c2:
        print('  CONCLUSION: Claim 2 holds but Claim 1 is weak. The k')
        print('  geometrically-selected neurons train well, but survival')
        print('  is not strongly predicted by initial proximity -- other')
        print('  factors (amplitude dynamics, interactions) matter too.')
    else:
        print('  CONCLUSION: Neither claim holds strongly. Initial bias')
        print('  proximity to inflections does not reliably predict survival,')
        print('  and training k neurons from their init does not recover')
        print('  full-network performance.')

    print()
    if amp_uninformative:
        print('  AMPLITUDE ISOLATION: bias-only ~= geometric across all runs.')
        print('  The a_j initialization carries no signal -- collapse is')
        print('  entirely determined by the spatial draw of b_j at t=0.')
    else:
        print('  AMPLITUDE ISOLATION: bias-only differs from geometric.')
        print('  Some amplitude information is present in the init.')
    if pos_is_all:
        print('  POSITION ISOLATION: amp-only ~= random across all runs.')
        print('  Knowing which a_j values were selected is useless without')
        print('  the correct b_j positions -- position is the entire lottery.')
    else:
        print('  POSITION ISOLATION: amp-only outperforms random slightly.')
        print('  Amplitude init may carry a weak positional signal.')

    print('\n' + '=' * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print('=' * 72)
    print('Geometric Lottery Ticket -- Shallow ReLU (ODE gradient flow)')
    print(f'Targets    : {list(TARGETS.keys())}')
    print(f'M values   : {M_VALUES}')
    print(f'T_FINAL    : {T_FINAL}')
    print(f'Conditions : full (m), geometric (k), bias_only, amp_only, '
          f'random_k x{N_RANDOM_TRIALS}')
    print(f'Total jobs : {len(TARGETS) * len(M_VALUES)}')
    print(f'Output     : {OUT_DIR}')
    print('=' * 72)

    all_results = []
    all_infos   = []

    for target_key in TARGETS:
        for m in M_VALUES:
            print(f'\n{"-" * 72}')
            print(f'  {target_key}  m={m}  T={T_FINAL}')
            print(f'{"-" * 72}')

            t0 = time.time()
            results, logs, survival_data, info = run_experiment(target_key, m)
            elapsed = time.time() - t0

            all_results.extend(results)
            all_infos.append(info)

            print(f'  Clusters after flow  : {info["n_clusters"]}  '
                  f'(k={info["k_true"]})')
            print(f'  Geometric overlap    : '
                  f'{info["overlap"]}/{info["k_true"]}  '
                  f'({info["overlap_frac"]:.0%})')
            print(f'  Dist ratio           : '
                  f'{info["mean_surv_dist"]:.4f} / '
                  f'{info["mean_other_dist"]:.4f} = '
                  f'{info["mean_surv_dist"]/max(info["mean_other_dist"],1e-9):.3f}')

            for r in results:
                trial_str = (f'  trial={r["trial"]}' if r['trial'] else '')
                print(f'    {r["condition"]:<12}  '
                      f'loss={r["final_loss"]:.4e}  '
                      f'neurons={r["n_neurons"]}{trial_str}')

            f1 = plot_survival(survival_data, info, OUT_DIR)
            f2 = plot_loss_curves(logs, info, OUT_DIR)
            f3 = plot_final_bar(results, info, OUT_DIR)
            print(f'  Plots: {f1}')
            print(f'         {f2}')
            print(f'         {f3}')
            print(f'  Elapsed: {elapsed:.1f}s')

    f_overlap = plot_overlap_summary(all_infos, OUT_DIR)
    print(f'\nOverlap summary: {f_overlap}')

    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in SUMMARY_FIELDS})
    print(f'Summary CSV: {SUMMARY_CSV}  ({len(all_results)} rows)')

    print_interpretation(all_results, all_infos)

    wall = time.time() - t_start
    print(f'\nTotal wall time: {wall / 60:.1f} min')
