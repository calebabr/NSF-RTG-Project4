"""
verify_pruning.py
=================
Numerically verifies the pruning bound from Open Problem 4.3 (slide 97):

    If intra-cluster diameter <= delta, then:

        || f_tilde - f ||_L2  <=  delta * sum_j |a_j|

where f is the converged network, f_tilde is the pruned network obtained by
replacing each cluster with a single neuron at the cluster centroid with the
summed amplitude.

Run this script AFTER simulate.py / simulate_parallel.py OR simulate_discrete.py
have completed. It scans every run folder under the selected figures directory for
convergence_check.csv and run_meta.csv. No re-simulation is needed.

Mode switch
-----------
  Set MODE = "flow"     to verify ODE flow results   (figures/Replication data/)
  Set MODE = "discrete" to verify discrete GD results (figures/Discrete GD/)

What this script does per run
------------------------------
  1. Read b_j, a_j from convergence_check.csv
  2. Group neurons into clusters (gap tolerance = 0.02, same as main scripts)
  3. Compute delta  = max intra-cluster diameter across all clusters
  4. Compute sum_abs_a = sum of |a_j| over all neurons
  5. Compute the bound RHS = delta * sum_abs_a
  6. Build the pruned network f_tilde (centroid bias, summed amplitude per cluster)
  7. Compute the actual pruning error || f_tilde - f ||_L2 on the quadrature grid
  8. Check: actual_error <= bound  (bound_holds)
  9. Compute tightness = actual_error / bound  (how close to tight the bound is)
 10. Also compute || f_tilde - f* ||_L2  (pruned network quality vs target)
 11. Save per-run figure and append to results CSV

Outputs
-------
  Per run (in each existing run folder):
      pruning_verification.png  -- full vs pruned vs target, pointwise error, bound check

  Global (in the selected figures directory):
      pruning_bound_results.csv -- one row per run, all metrics
      pruning_bound_summary.png -- scatter and trend plots across all runs

Usage
-----
    python verify_pruning.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# ── Mode switch ───────────────────────────────────────────────────────────────
# "flow"             → ODE results in figures/Replication data/
# "discrete"         → constant-lr GD in figures/Discrete GD/
# "discrete_scaled"  → scaled-lr GD in figures/Discrete GD Scaled/
# =============================================================================
MODE = "flow"
# "flow"     → ODE results in figures/Replication data/
# "discrete" → GD results  in figures/Discrete GD/

if MODE == "flow":
    FIG_BASE   = os.path.join('figures', 'Replication data')
    TIME_FIELD = 'T'        # field name in run_meta.csv
    TIME_LABEL = 'T'        # used in figure titles and print output
else:
    FIG_BASE   = os.path.join('figures', 'Discrete GD')
    TIME_FIELD = 'steps'
    TIME_LABEL = 'steps'

RESULTS_CSV = os.path.join(FIG_BASE, 'pruning_bound_results.csv')
SUMMARY_PNG = os.path.join(FIG_BASE, 'pruning_bound_summary.png')

# =============================================================================
# Quadrature grid  (must match main scripts)
# =============================================================================
N_QUAD = 400
X_QUAD = np.linspace(-1.0, 1.0, N_QUAD)
DX     = X_QUAD[1] - X_QUAD[0]

# =============================================================================
# Target functions  (re-defined here so this script is self-contained)
# =============================================================================
TARGETS = {
    'sin_1pi': (r'$\sin(\pi x)$',      1, lambda x: np.sin(    np.pi * x)),
    'x_cubed': (r'$x^3$',              1, lambda x: x**3),
    'sin_2pi': (r'$\sin(2\pi x)$',     3, lambda x: np.sin(2 * np.pi * x)),
    'poly_k3': (r'$x^5-3x^3$',         3, lambda x: x**5 - 3 * x**3),
    'sin_3pi': (r'$\sin(3\pi x)$',     5, lambda x: np.sin(3 * np.pi * x)),
    'sin_4pi': (r'$\sin(4\pi x)$',     7, lambda x: np.sin(4 * np.pi * x)),
    'sin_5pi': (r'$\sin(5\pi x)$',     9, lambda x: np.sin(5 * np.pi * x)),
    'sin_6pi': (r'$\sin(6\pi x)$',    11, lambda x: np.sin(6 * np.pi * x)),
    'sin_7pi': (r'$\sin(7\pi x)$',    13, lambda x: np.sin(7 * np.pi * x)),
}

CLUSTER_TOL = 0.02   # same tolerance used in simulate.py / simulate_parallel.py

RESULTS_FIELDS = [
    'target', 'm', 'T', 'k_true', 'n_clusters',
    'delta',            # max intra-cluster diameter
    'sum_abs_a',        # sum of |a_j| over all neurons
    'bound',            # delta * sum_abs_a  (RHS of the pruning inequality)
    'actual_error',     # || f_tilde - f ||_L2  (LHS)
    'bound_holds',      # 1 if actual_error <= bound
    'tightness',        # actual_error / bound  (1.0 = tight, <1 = slack)
    'pruned_vs_target', # || f_tilde - f* ||_L2
    'full_vs_target',   # || f       - f* ||_L2  (from run_meta loss * 2, approx)
]

# =============================================================================
# Core math helpers
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def eval_network(x, a, b):
    """f(x) = sum_j a_j * relu(x - b_j)."""
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def l2_norm(f1, f2):
    """|| f1 - f2 ||_L2 on X_QUAD."""
    return float(np.sqrt(np.trapezoid((f1 - f2)**2, X_QUAD)))

# =============================================================================
# Cluster analysis
# =============================================================================

def identify_clusters(b_sorted, a_sorted, tol=CLUSTER_TOL):
    """
    Given b_j and a_j sorted by bias location, return a list of dicts:
        {'center': centroid, 'amp_sum': sum(a_j), 'diameter': max-min b_j,
         'indices': list of positions in the sorted array}
    """
    clusters = []
    group_b, group_a, group_idx = [b_sorted[0]], [a_sorted[0]], [0]

    for i in range(1, len(b_sorted)):
        if b_sorted[i] - b_sorted[i - 1] > tol:
            clusters.append({
                'center':   float(np.mean(group_b)),
                'amp_sum':  float(np.sum(group_a)),
                'diameter': float(max(group_b) - min(group_b)),
                'indices':  group_idx,
            })
            group_b, group_a, group_idx = [], [], []
        group_b.append(b_sorted[i])
        group_a.append(a_sorted[i])
        group_idx.append(i)

    clusters.append({
        'center':   float(np.mean(group_b)),
        'amp_sum':  float(np.sum(group_a)),
        'diameter': float(max(group_b) - min(group_b)),
        'indices':  group_idx,
    })
    return clusters

def build_pruned_network(clusters):
    """Return (b_pruned, a_pruned) arrays for the pruned network f_tilde."""
    b_p = np.array([c['center']  for c in clusters])
    a_p = np.array([c['amp_sum'] for c in clusters])
    return b_p, a_p

# =============================================================================
# Load a single run's data from convergence_check.csv + run_meta.csv
# =============================================================================

def load_run(run_dir):
    """
    Returns dict with keys: target, m, T, k_true, n_clusters, b, a
    or None if required files are missing.
    The 'T' key stores ODE end-time (flow mode) or step count (discrete mode).
    """
    f_csv  = os.path.join(run_dir, 'convergence_check.csv')
    f_meta = os.path.join(run_dir, 'run_meta.csv')
    if not (os.path.exists(f_csv) and os.path.exists(f_meta)):
        return None

    # Metadata
    with open(f_meta, newline='') as mf:
        meta = next(csv.DictReader(mf))

    # Per-neuron data — already sorted by b_j
    b_vals, a_vals = [], []
    with open(f_csv, newline='') as cf:
        for row in csv.DictReader(cf):
            b_vals.append(float(row['b_j']))
            a_vals.append(float(row['a_j']))

    return {
        'target':     meta['target'],
        'm':          int(meta['m']),
        'T':          int(meta[TIME_FIELD]),   # T or steps depending on mode
        'k_true':     int(meta['k_true']),
        'n_clusters': int(meta['n_clusters']),
        'b':          np.array(b_vals),
        'a':          np.array(a_vals),
    }

# =============================================================================
# Per-run verification and figure
# =============================================================================

def verify_one(run_dir, run_data):
    """
    Verify the pruning bound for one run. Saves pruning_verification.png.
    Returns a result dict for the CSV, or None on error.
    """
    target_key = run_data['target']
    m          = run_data['m']
    T          = run_data['T']
    k_true     = run_data['k_true']
    b          = run_data['b']
    a          = run_data['a']

    if target_key not in TARGETS:
        return None
    target_label, _, f_star = TARGETS[target_key]

    # ── Cluster analysis ──────────────────────────────────────────────────────
    clusters   = identify_clusters(b, a)
    n_clusters = len(clusters)
    delta      = max(c['diameter'] for c in clusters)
    sum_abs_a  = float(np.abs(a).sum())
    bound      = delta * sum_abs_a

    # ── Build networks on quadrature grid ─────────────────────────────────────
    f_full   = eval_network(X_QUAD, a, b)
    b_p, a_p = build_pruned_network(clusters)
    f_pruned = eval_network(X_QUAD, a_p, b_p)
    f_target = f_star(X_QUAD)

    # ── Compute norms ─────────────────────────────────────────────────────────
    actual_error     = l2_norm(f_pruned, f_full)
    pruned_vs_target = l2_norm(f_pruned, f_target)
    full_vs_target   = l2_norm(f_full,   f_target)
    bound_holds      = int(actual_error <= bound + 1e-12)
    tightness        = actual_error / bound if bound > 1e-15 else float('nan')

    # ── Figure ────────────────────────────────────────────────────────────────
    out_fig = os.path.join(run_dir, 'pruning_verification.png')
    if not os.path.exists(out_fig):
        x_plot   = np.linspace(-1.0, 1.0, 500)
        ff_plot  = eval_network(x_plot, a,   b)
        fp_plot  = eval_network(x_plot, a_p, b_p)
        ft_plot  = f_star(x_plot)
        diff_plot = np.abs(fp_plot - ff_plot)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Open Problem 4.3 — Pruning Bound Verification\n'
            f'target={target_label},  m={m},  {TIME_LABEL}={T}  |  '
            f'clusters={n_clusters},  k={k_true}',
            fontsize=11)

        # Panel 1: full vs pruned vs target
        ax = axes[0]
        ax.plot(x_plot, ft_plot,  'k--', lw=2,   label='Target $f^*$')
        ax.plot(x_plot, ff_plot,  'steelblue', lw=1.5, label=f'Full $f$  ({m} neurons)')
        ax.plot(x_plot, fp_plot,  'r-',  lw=2,   label=f'Pruned $\\tilde{{f}}$  ({n_clusters} neurons)')
        ax.set_xlabel('$x$')
        ax.set_title('Full vs Pruned vs Target')
        ax.legend(fontsize=8)
        ax.set_xlim([-1, 1])
        ax.grid(True, alpha=0.3)

        # Panel 2: pointwise pruning error |f_tilde - f|
        ax = axes[1]
        ax.plot(x_plot, diff_plot, color='darkorange', lw=1.5)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|\\tilde{f}(x) - f(x)|$')
        ax.set_title('Pointwise Pruning Error')
        ax.grid(True, alpha=0.3)

        # Panel 3: bound check bar chart
        ax = axes[2]
        bars = ax.bar(['Actual error\n$\\|\\tilde{f}-f\\|_{L^2}$',
                       'Bound\n$\\delta \\cdot \\sum|a_j|$'],
                      [actual_error, bound],
                      color=['steelblue' if bound_holds else 'crimson', 'lightgray'],
                      edgecolor='black', linewidth=0.8)
        ax.set_ylabel('Value')
        status = 'HOLDS' if bound_holds else 'VIOLATED'
        color  = 'green' if bound_holds else 'red'
        ax.set_title(
            f'Bound {status}  (tightness={tightness:.3f})\n'
            f'$\\delta$={delta:.4f},  $\\sum|a_j|$={sum_abs_a:.2f}',
            color=color, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches='tight')
        plt.close()

    return {
        'target':           target_key,
        'm':                m,
        'T':                T,
        'k_true':           k_true,
        'n_clusters':       n_clusters,
        'delta':            f'{delta:.6e}',
        'sum_abs_a':        f'{sum_abs_a:.6f}',
        'bound':            f'{bound:.6e}',
        'actual_error':     f'{actual_error:.6e}',
        'bound_holds':      bound_holds,
        'tightness':        f'{tightness:.6f}' if not np.isnan(tightness) else 'nan',
        'pruned_vs_target': f'{pruned_vs_target:.6e}',
        'full_vs_target':   f'{full_vs_target:.6e}',
    }

# =============================================================================
# Summary figure
# =============================================================================

def make_summary_figure(results):
    """Four-panel summary across all runs."""
    # Parse numeric columns
    def col(key):
        vals = []
        for r in results:
            try: vals.append(float(r[key]))
            except: vals.append(float('nan'))
        return np.array(vals)

    actual  = col('actual_error')
    bound   = col('bound')
    tight   = col('tightness')
    sum_a   = col('sum_abs_a')
    delta   = col('delta')
    ms      = np.array([int(r['m']) for r in results])
    holds   = np.array([int(r['bound_holds']) for r in results])

    # Color by target
    target_keys = list(dict.fromkeys(r['target'] for r in results))
    cmap        = plt.cm.tab10(np.linspace(0, 0.85, len(target_keys)))
    colors      = np.array([cmap[target_keys.index(r['target'])] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Open Problem 4.3 — Pruning Bound Summary', fontsize=14)

    # Panel 1: actual error vs bound (scatter — should all be below y=x)
    ax = axes[0, 0]
    for ti, tkey in enumerate(target_keys):
        mask = [r['target'] == tkey for r in results]
        ax.scatter(bound[mask], actual[mask], color=cmap[ti], s=30,
                   label=tkey, zorder=3)
    lim = max(np.nanmax(bound), np.nanmax(actual)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, label='actual = bound')
    ax.set_xlabel('Bound  $\\delta \\cdot \\sum|a_j|$')
    ax.set_ylabel('Actual error  $\\|\\tilde{f}-f\\|_{L^2}$')
    ax.set_title(f'Bound check  ({holds.sum()}/{len(holds)} hold)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: tightness vs m (how close to saturating the bound)
    ax = axes[0, 1]
    for ti, tkey in enumerate(target_keys):
        mask  = np.array([r['target'] == tkey for r in results])
        order = np.argsort(ms[mask])
        ax.plot(ms[mask][order], tight[mask][order],
                marker='o', ms=4, color=cmap[ti], lw=1.2, label=tkey)
    ax.axhline(1.0, color='k', lw=1, linestyle='--', label='tight (=1)')
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Tightness  (actual / bound)')
    ax.set_title('Bound Tightness vs m')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: sum |a_j| vs m (the amplitude control challenge)
    ax = axes[1, 0]
    for ti, tkey in enumerate(target_keys):
        mask  = np.array([r['target'] == tkey for r in results])
        order = np.argsort(ms[mask])
        ax.plot(ms[mask][order], sum_a[mask][order],
                marker='s', ms=4, color=cmap[ti], lw=1.2, label=tkey)
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('$\\sum_j |a_j|$')
    ax.set_title('Amplitude Sum vs m\n(key challenge: does this stay bounded?)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: delta vs m (how tight the clusters are)
    ax = axes[1, 1]
    for ti, tkey in enumerate(target_keys):
        mask  = np.array([r['target'] == tkey for r in results])
        order = np.argsort(ms[mask])
        ax.semilogy(ms[mask][order], delta[mask][order],
                    marker='^', ms=4, color=cmap[ti], lw=1.2, label=tkey)
    ax.set_xlabel('Width $m$')
    ax.set_ylabel('$\\delta$ (max intra-cluster diameter)')
    ax.set_title('Cluster Tightness vs m\n(smaller = more merged)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Summary figure -> {SUMMARY_PNG}')

# =============================================================================
# Worker wrapper  (module-level so multiprocessing can pickle it on Windows)
# run_data contains only strings/ints/numpy arrays — no lambdas passed through.
# TARGETS is looked up inside verify_one after the worker process re-imports
# this module, so the lambdas never need to be pickled.
# =============================================================================

def _worker(args):
    run_dir, run_data = args
    return verify_one(run_dir, run_data)

# =============================================================================
# Main — scan all run folders and verify in parallel
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()

    # ── Discover all run folders ───────────────────────────────────────────────
    run_dirs = []
    for target_key in os.listdir(FIG_BASE):
        target_path = os.path.join(FIG_BASE, target_key)
        if not os.path.isdir(target_path) or target_key not in TARGETS:
            continue
        for m_dir in os.listdir(target_path):
            m_path = os.path.join(target_path, m_dir)
            if not os.path.isdir(m_path):
                continue
            for t_dir in os.listdir(m_path):
                if t_dir != 'T=500':
                    continue
                t_path = os.path.join(m_path, t_dir)
                if os.path.isdir(t_path):
                    run_dirs.append(t_path)

    run_dirs.sort()
    print(f'Mode: {MODE}  →  {FIG_BASE}')
    print(f'Found {len(run_dirs)} run folders')

    # ── Load existing results to support restart ───────────────────────────────
    existing = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['target'], int(row['m']), int(row['T']))
                existing[key] = row
        print(f'Loaded {len(existing)} existing results from {RESULTS_CSV}')

    # ── Build job list, skip already completed runs ────────────────────────────
    jobs       = []
    skip_count = 0

    for run_dir in run_dirs:
        run_data = load_run(run_dir)
        if run_data is None:
            continue
        key     = (run_data['target'], run_data['m'], run_data['T'])
        out_fig = os.path.join(run_dir, 'pruning_verification.png')
        if key in existing and os.path.exists(out_fig):
            skip_count += 1
            print(f'  SKIP  {run_data["target"]:<12}  m={run_data["m"]:<5}  {TIME_LABEL}={run_data["T"]}')
        else:
            jobs.append((run_dir, run_data))

    n_workers = max(1, cpu_count() - 2)
    print(f'\nJobs to run: {len(jobs)}   Workers: {n_workers}   Skipped: {skip_count}')

    # ── Run in parallel ────────────────────────────────────────────────────────
    all_results = dict(existing)
    new_count   = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker, jobs):
            if result is None:
                continue
            key    = (result['target'], int(result['m']), int(result['T']))
            all_results[key] = result
            new_count += 1
            status = 'HOLDS' if result['bound_holds'] else '*** VIOLATED ***'
            print(f'  DONE  {result["target"]:<12}  m={result["m"]:<5}  {TIME_LABEL}={result["T"]:<6}  '
                  f'actual={float(result["actual_error"]):.3e}  '
                  f'bound={float(result["bound"]):.3e}  '
                  f'tight={result["tightness"]}  '
                  f'[{status}]')

    # ── Write results CSV ──────────────────────────────────────────────────────
    sorted_results = sorted(all_results.values(),
                            key=lambda r: (r['target'], int(r['m']), int(r['T'])))
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow({k: r[k] for k in RESULTS_FIELDS})
    print(f'\nResults CSV -> {RESULTS_CSV}  ({len(sorted_results)} rows)')

    # ── Summary figure ─────────────────────────────────────────────────────────
    if sorted_results:
        make_summary_figure(sorted_results)

    wall = time.time() - t_start
    print(f'\nDone.  {new_count} new,  {skip_count} skipped,  wall time {wall:.1f}s')
