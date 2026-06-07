"""
simulate_parallel.py
====================
Parallel version of simulate.py using multiprocessing.Pool.

Conjecture (Open Problem 4.1, slide 95):
    lim_{m -> inf} C(m, f*) = #{x in [-1,1] : f*''(x) = 0 and changes sign}

New in this script vs simulate.py
----------------------------------
  - Multiprocessing: runs multiple (target, m, T) combos simultaneously
  - Larger m : [2000, 3000, 5000]  (in addition to existing [50..1500])
  - T values : [5000, 10000] only  (lower T gave no convergence evidence)
  - Three new targets with k >= 9:
      sin_5pi  sin(5*pi*x)  k=9
      sin_6pi  sin(6*pi*x)  k=11
      sin_7pi  sin(7*pi*x)  k=13

Skip logic
----------
  All (target, m, T) combinations are new since T in {5000, 10000} was never
  run by simulate.py. Skip only if all 4 output files + run_meta.csv already
  exist (Ctrl+C safety across restarts).

Output files  — does NOT overwrite simulate.py outputs
-------------------------------------------------------
  Per run  : figures/Replication data/{target}/m={m}/T={T}/  (4 files + run_meta)
  Summary  : figures/Replication data/run_summary_parallel.csv
  Plot     : figures/Replication data/convergence_plot_parallel.png
             (includes simulate.py data from run_summary.csv for full picture)

Usage
-----
    python simulate_parallel.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Quadrature grid
# =============================================================================
N_QUAD = 200
X_QUAD = np.linspace(-1.0, 1.0, N_QUAD)
DX     = X_QUAD[1] - X_QUAD[0]

# =============================================================================
# Core math  (module-level so workers can access after re-import on Windows)
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

def compute_ode_velocities(a, b, f_star):
    fstar_vals = f_star(X_QUAD)
    residual   = network(X_QUAD, a, b) - fstar_vals
    relu_mat   = relu(X_QUAD[:, None] - b[None, :])
    da         = -(residual[:, None] * relu_mat).sum(0) * DX
    cum_right  = np.cumsum(residual[::-1])[::-1] * DX
    idx        = np.searchsorted(X_QUAD, b).clip(0, N_QUAD - 1)
    R          = cum_right[idx]
    db         = a * R
    return da, db, R

def count_clusters(biases, tol=0.02):
    s = np.sort(biases)
    return 1 + int(np.sum(np.diff(s) > tol))

def get_cluster_centers(biases, tol=0.02):
    s = np.sort(biases)
    centers, group = [], [s[0]]
    for i in range(1, len(s)):
        if s[i] - s[i - 1] > tol:
            centers.append(float(np.mean(group)))
            group = []
        group.append(s[i])
    centers.append(float(np.mean(group)))
    return np.array(centers)

# =============================================================================
# Target functions — must be named defs (not lambdas) for multiprocessing pickle
#
# Analytical inflection-point counts for sin(n*pi*x):
#   f''(x) = -n^2*pi^2 * sin(n*pi*x)
#   zeros at x = j/n for j in {-(n-1), ..., 0, ..., n-1}  => 2n-1 inflection pts
# =============================================================================

def f_sin_1pi(x): return np.sin(    np.pi * x)
def f_x_cubed(x): return x**3
def f_sin_2pi(x): return np.sin(2 * np.pi * x)
def f_poly_k3(x): return x**5 - 3 * x**3
def f_sin_3pi(x): return np.sin(3 * np.pi * x)
def f_sin_4pi(x): return np.sin(4 * np.pi * x)
def f_sin_5pi(x): return np.sin(5 * np.pi * x)   # k=9
def f_sin_6pi(x): return np.sin(6 * np.pi * x)   # k=11
def f_sin_7pi(x): return np.sin(7 * np.pi * x)   # k=13

# key -> (plot label, analytical k, function)
TARGETS = {
    'sin_1pi': (r'$\sin(\pi x)$',      1,  f_sin_1pi),
    'x_cubed': (r'$x^3$',              1,  f_x_cubed),
    'sin_2pi': (r'$\sin(2\pi x)$',     3,  f_sin_2pi),
    'poly_k3': (r'$x^5-3x^3$',         3,  f_poly_k3),
    'sin_3pi': (r'$\sin(3\pi x)$',     5,  f_sin_3pi),
    'sin_4pi': (r'$\sin(4\pi x)$',     7,  f_sin_4pi),
    'sin_5pi': (r'$\sin(5\pi x)$',     9,  f_sin_5pi),
    'sin_6pi': (r'$\sin(6\pi x)$',    11,  f_sin_6pi),
    'sin_7pi': (r'$\sin(7\pi x)$',    13,  f_sin_7pi),
}

def get_inflection_locations(target_key):
    locs = {
        'sin_1pi': [0.0],
        'x_cubed': [0.0],
        'sin_2pi': [-0.5, 0.0, 0.5],
        'poly_k3': [-np.sqrt(0.9), 0.0, np.sqrt(0.9)],
        'sin_3pi': [-2/3, -1/3, 0.0, 1/3, 2/3],
        'sin_4pi': [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
        'sin_5pi': [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        'sin_6pi': [-5/6, -4/6, -3/6, -2/6, -1/6, 0.0,
                     1/6,  2/6,  3/6,  4/6,  5/6],
        'sin_7pi': [-6/7, -5/7, -4/7, -3/7, -2/7, -1/7, 0.0,
                     1/7,  2/7,  3/7,  4/7,  5/7,  6/7],
    }
    return np.array(locs.get(target_key, []))

# =============================================================================
# Sweep parameters
# =============================================================================

M_VALUES_ALL = [50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000]
T_VALUES_ALL = [5000, 10000]   # only long runs — lower T gave no convergence evidence

N_SAVE           = 300
SEED             = 42
ACTIVE_THRESHOLD = 0.05

FIG_BASE         = os.path.join('figures', 'Replication data')
ORIG_SUMMARY_CSV = os.path.join(FIG_BASE, 'run_summary.csv')          # simulate.py — read only
SUMMARY_CSV      = os.path.join(FIG_BASE, 'run_summary_parallel.csv') # this script's output only
CONV_PLOT        = os.path.join(FIG_BASE, 'convergence_plot_parallel.png')

META_FIELDS = ['target', 'm', 'T', 'k_true', 'n_clusters', 'n_active',
               'loss', 'max_da', 'max_db', 'active_leq_k']

# =============================================================================
# Single-run worker  (called by each pool process)
# =============================================================================

def run_one(args):
    """
    Simulate one (target_key, m, T) combination and write all outputs.
    Returns a metrics dict (with '_skipped' flag if recovered from run_meta.csv),
    or None if the run was already done before run_meta.csv existed.
    """
    target_key, m, T = args
    target_label, k_true, f_star = TARGETS[target_key]

    out_dir = os.path.join(FIG_BASE, target_key, f'm={m}', f'T={T}')
    f_traj  = os.path.join(out_dir, 'slide93_reproduction.png')
    f_clust = os.path.join(out_dir, 'clusters_vs_inflections.png')
    f_ode   = os.path.join(out_dir, 'ode_verification.png')
    f_csv   = os.path.join(out_dir, 'convergence_check.csv')
    f_meta  = os.path.join(out_dir, 'run_meta.csv')

    # All output files exist — recover metrics from run_meta.csv
    if all(os.path.exists(p) for p in [f_traj, f_clust, f_ode, f_csv]):
        if os.path.exists(f_meta):
            with open(f_meta, newline='') as mf:
                row = next(csv.DictReader(mf))
            return {
                'target': target_key, 'm': m, 'T': T,
                'k_true': int(row['k_true']),
                'n_clusters': int(row['n_clusters']),
                'n_active': int(row['n_active']),
                'loss': float(row['loss']),
                'max_da': float(row['max_da']),
                'max_db': float(row['max_db']),
                'active_leq_k': int(row['active_leq_k']),
                '_skipped': True,
            }
        return None  # pre-run_meta.csv era, skip silently

    os.makedirs(out_dir, exist_ok=True)

    # ── Simulate ──────────────────────────────────────────────────────────────
    t0 = time.time()
    np.random.seed(SEED)
    a0 = np.random.randn(m) * 0.01
    b0 = np.random.uniform(-1.0, 1.0, m)

    sol = solve_ivp(
        make_ode(m, f_star),
        t_span=(0.0, T),
        y0=np.concatenate([a0, b0]),
        method='RK45',
        t_eval=np.linspace(0.0, T, N_SAVE),
        rtol=1e-4, atol=1e-6,
        max_step=max(0.1, T / 500),
    )
    elapsed = time.time() - t0

    # ── Loss (sparse) ─────────────────────────────────────────────────────────
    fstar_vals = f_star(X_QUAD)
    snap_idx   = np.arange(0, N_SAVE, 5)
    losses_sparse = np.array([
        0.5 * np.trapezoid(
            (network(X_QUAD, sol.y[:m, i], sol.y[m:, i]) - fstar_vals)**2, X_QUAD)
        for i in snap_idx
    ])
    t_sparse   = sol.t[snap_idx]
    final_loss = float(0.5 * np.trapezoid(
        (network(X_QUAD, sol.y[:m, -1], sol.y[m:, -1]) - fstar_vals)**2, X_QUAD))

    # ── Final state ───────────────────────────────────────────────────────────
    a_final         = sol.y[:m, -1]
    b_final         = sol.y[m:, -1]
    x_plot          = np.linspace(-1.0, 1.0, 500)
    f_final         = network(x_plot, a_final, b_final)
    n_clusters      = count_clusters(b_final)
    cluster_centers = get_cluster_centers(b_final)

    # ── Goal 3: stationarity ──────────────────────────────────────────────────
    da_final, db_final, R_final = compute_ode_velocities(a_final, b_final, f_star)
    a_max     = max(float(np.abs(a_final).max()), 1e-12)
    is_active = np.abs(a_final) > ACTIVE_THRESHOLD * a_max
    n_active  = int(is_active.sum())

    sort_idx = np.argsort(b_final)
    b_s      = b_final[sort_idx];  a_s = a_final[sort_idx]
    da_s     = da_final[sort_idx]; db_s = db_final[sort_idx]
    R_s      = R_final[sort_idx];  active_s = is_active[sort_idx]

    infl_x = get_inflection_locations(target_key)

    # ── Figure 1: trajectories / fit / loss ───────────────────────────────────
    if not os.path.exists(f_traj):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'target={target_label},  m={m},  T={T}'
            f'  |  clusters={n_clusters},  k={k_true}', fontsize=12)

        ax = axes[0]
        sorted_b_traj = np.sort(sol.y[m:, :], axis=0)
        alpha = min(0.4, 20.0 / m)
        for j in range(m):
            ax.plot(sol.t, sorted_b_traj[j], color='steelblue',
                    alpha=alpha, linewidth=0.4)
        ax.set_xlabel('Time'); ax.set_ylabel('Bias value')
        ax.set_title('Sorted Bias Trajectories'); ax.set_xlim([0, T])

        ax = axes[1]
        ax.plot(x_plot, f_star(x_plot), 'k--', lw=2, label=f'Target {target_label}')
        ax.plot(x_plot, f_final, 'r-', lw=2, label='Network $f$')
        # Show cluster centers as tick marks instead of all m bias dots
        y0, y1  = ax.get_ylim()
        tick_h  = (y1 - y0) * 0.06
        ax.vlines(cluster_centers, -tick_h/2, tick_h/2,
                  colors='steelblue', linewidths=2.5, zorder=5)
        ax.set_xlabel('$x$'); ax.set_title('Final Fit vs Target')
        ax.legend(fontsize=8); ax.set_xlim([-1, 1])

        ax = axes[2]
        ax.semilogy(t_sparse, losses_sparse, color='darkorange', lw=1.5)
        ax.set_xlabel('Time'); ax.set_ylabel('MSE Loss')
        ax.set_title(f'Loss (log)   final={final_loss:.2e}')
        ax.set_xlim([0, T]); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f_traj, bbox_inches='tight')
        plt.close()

    # ── Figure 2: clusters vs inflection points ────────────────────────────────
    if not os.path.exists(f_clust):
        x_fine = np.linspace(-1.0, 1.0, 2000)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(
            f'target={target_label},  m={m},  T={T}  '
            f'|  clusters={n_clusters},  k={k_true}  '
            f'(C=k: {n_clusters == k_true})', fontsize=11)
        ax.plot(x_fine, f_star(x_fine), 'k-', lw=2, label=f'Target {target_label}')
        ax.plot(x_plot, f_final, 'r-', lw=1.5, label='Final network')
        for cx in cluster_centers:
            ax.axvline(cx, color='steelblue', alpha=0.5, lw=0.7)
        for ix in infl_x:
            ax.axvline(ix, color='green', alpha=0.7, lw=1.2, linestyle='--')
        from matplotlib.lines import Line2D
        h, _ = ax.get_legend_handles_labels()
        h += [
            Line2D([0],[0], color='steelblue', lw=1.5,
                   label=f'Bias clusters ({n_clusters})'),
            Line2D([0],[0], color='green', lw=1.5, ls='--',
                   label=f'Inflection pts k={k_true}'),
        ]
        ax.legend(handles=h, fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_title('Cluster Locations vs Inflection Points')
        plt.tight_layout()
        plt.savefig(f_clust, bbox_inches='tight')
        plt.close()

    # ── Figure 3: ODE stationarity check ──────────────────────────────────────
    if not os.path.exists(f_ode):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Stationarity — target={target_label},  m={m},  T={T}\n'
            f'active={n_active},  k={k_true},  active<=k: {n_active<=k_true}',
            fontsize=10)

        ax = axes[0]
        ax.scatter(b_s, np.abs(da_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
        ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{a}_j|$')
        ax.set_title('Amplitude velocity  (should be $\\approx 0$)')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(b_s, np.abs(db_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
        ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{b}_j|$')
        ax.set_title('Bias velocity  (should be $\\approx 0$)')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        ax = axes[2]
        sizes = np.clip(np.abs(a_s) / a_max * 80, 4, 80)
        sc = ax.scatter(b_s, R_s, c=np.abs(a_s), cmap='viridis', s=sizes)
        ax.scatter(b_s[active_s], R_s[active_s],
                   edgecolors='red', facecolors='none', s=60, lw=1.2,
                   label=f'Active ({n_active})')
        ax.axhline(0.0, color='k', lw=1.0, linestyle='--')
        plt.colorbar(sc, ax=ax, label='$|a_j|$')
        ax.set_xlabel('$b_j$')
        ax.set_ylabel('$R_j = \\int_{b_j}^{1}(f-f^*)\\,dx$')
        ax.set_title(f'Integrated residual  active={n_active} <= k={k_true}: {n_active<=k_true}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f_ode, bbox_inches='tight')
        plt.close()

    # ── CSV: per-neuron stationarity data ─────────────────────────────────────
    if not os.path.exists(f_csv):
        R_tol = 1e-3
        with open(f_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['neuron_idx', 'b_j', 'a_j',
                             'da_dt', 'db_dt', 'R_j',
                             'is_active', 'R_near_zero'])
            for i in range(m):
                writer.writerow([
                    sort_idx[i],
                    f'{b_s[i]:.6f}',  f'{a_s[i]:.6f}',
                    f'{da_s[i]:.6e}', f'{db_s[i]:.6e}', f'{R_s[i]:.6e}',
                    int(active_s[i]), int(abs(R_s[i]) < R_tol),
                ])

    # ── run_meta.csv: single-row summary for skip recovery ────────────────────
    result = {
        'target': target_key, 'm': m, 'T': T,
        'k_true': k_true, 'n_clusters': n_clusters,
        'n_active': n_active, 'loss': final_loss,
        'max_da': float(np.abs(da_final).max()),
        'max_db': float(np.abs(db_final).max()),
        'active_leq_k': int(n_active <= k_true),
    }
    with open(f_meta, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerow({k: result[k] for k in META_FIELDS})

    result['_elapsed'] = round(elapsed, 1)
    return result

# =============================================================================
# Convergence plot
# =============================================================================

def make_convergence_plot(summary_rows, out_path=None):
    targets_present = list(dict.fromkeys(r['target'] for r in summary_rows))
    T_present       = sorted(set(r['T'] for r in summary_rows))
    markers         = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    cmap            = plt.cm.tab10(np.linspace(0, 0.8, len(T_present)))

    n_t   = len(targets_present)
    ncols = 3
    nrows = -(-n_t // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, tkey in enumerate(targets_present):
        ax     = axes_flat[ax_i]
        rows_t = [r for r in summary_rows if r['target'] == tkey]
        k_true = rows_t[0]['k_true']
        label  = TARGETS[tkey][0] if tkey in TARGETS else tkey

        for ti, T in enumerate(T_present):
            rows_mT = sorted([r for r in rows_t if r['T'] == T],
                             key=lambda r: r['m'])
            if not rows_mT:
                continue
            ms = [r['m']         for r in rows_mT]
            cs = [r['n_clusters'] for r in rows_mT]
            ax.plot(ms, cs, marker=markers[ti % len(markers)],
                    color=cmap[ti], lw=1.5, ms=5, label=f'T={T}')

        ax.axhline(k_true, color='crimson', lw=2, linestyle='--',
                   label=f'k={k_true}')
        ax.set_xlabel('Width $m$')
        ax.set_ylabel('Cluster count $C(m,f^*)$')
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax_i in range(len(targets_present), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(
        'Open Problem 4.1 — $C(m,f^*) \\to k$ as $m \\to \\infty$\n'
        'Crimson dashed = analytical inflection-point count $k$',
        fontsize=13)
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(FIG_BASE, 'convergence_plot_parallel.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Convergence plot -> {out_path}')

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(FIG_BASE, exist_ok=True)

    # ── Build job list ─────────────────────────────────────────────────────────
    # All T values here are new (5000, 10000 were never run by simulate.py),
    # so no old-combo filtering needed — skip logic lives inside run_one via
    # run_meta.csv for Ctrl+C safety only.
    jobs = [(t, m, T)
            for t in TARGETS
            for m in M_VALUES_ALL
            for T in T_VALUES_ALL]

    # ── Load simulate.py rows (read-only) for the combined convergence plot ───
    def load_csv(path):
        rows = []
        if os.path.exists(path):
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    rows.append({
                        'target': row['target'],
                        'm':      int(row['m']),
                        'T':      int(row['T']),
                        'k_true': int(row['k_true']),
                        'n_clusters': int(row['n_clusters']),
                        'n_active':   int(row['n_active']),
                        'loss':       float(row['loss']),
                        'max_da':     float(row['max_da']),
                        'max_db':     float(row['max_db']),
                        'active_leq_k': int(row['active_leq_k']),
                    })
        return rows

    orig_rows     = load_csv(ORIG_SUMMARY_CSV)    # simulate.py rows — never written to
    existing_rows = load_csv(SUMMARY_CSV)         # previous parallel runs (T=5000,10000 only)

    n_workers = max(1, cpu_count() - 2)

    print('=' * 72)
    print('simulate_parallel.py — Open Problem 4.1')
    print(f'Targets  : {list(TARGETS.keys())}')
    print(f'm values : {M_VALUES_ALL}')
    print(f'T values : {T_VALUES_ALL}')
    print(f'Jobs     : {len(jobs)}')
    print(f'Workers  : {n_workers}  (of {cpu_count()} logical CPUs)')
    print(f'simulate.py rows (plot only) : {len(orig_rows)}')
    print(f'Previous parallel rows       : {len(existing_rows)}')
    print(f'Summary  -> {SUMMARY_CSV}')
    print(f'Plot     -> {CONV_PLOT}')
    print('=' * 72)

    # ── Run in parallel ────────────────────────────────────────────────────────
    new_rows  = []
    skipped   = 0
    completed = 0
    t_start   = time.time()

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            if result is None:
                skipped += 1
                continue

            was_skipped = result.pop('_skipped', False)
            elapsed     = result.pop('_elapsed', None)
            new_rows.append(result)

            tag = (f'{result["target"]:<12}  m={result["m"]:<5}  T={result["T"]:<6}  '
                   f'clusters={result["n_clusters"]:<4}  k={result["k_true"]}  '
                   f'C=k: {result["n_clusters"]==result["k_true"]}  '
                   f'loss={result["loss"]:.3e}  '
                   f'max|db|={result["max_db"]:.2e}')

            if was_skipped:
                skipped += 1
                print(f'  SKIP (meta)  {tag}')
            else:
                completed += 1
                t_str = f'  [{elapsed:.0f}s]' if elapsed else ''
                print(f'  DONE{t_str}  {tag}')

    # ── Write run_summary_parallel.csv (parallel runs only) ───────────────────
    parallel_rows = {(r['target'], r['m'], r['T']): r for r in existing_rows}
    for r in new_rows:
        parallel_rows[(r['target'], r['m'], r['T'])] = r
    parallel_rows = sorted(parallel_rows.values(),
                           key=lambda r: (r['target'], r['m'], r['T']))

    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for r in parallel_rows:
            writer.writerow({k: r[k] for k in META_FIELDS})
    print(f'\nParallel summary -> {SUMMARY_CSV}  ({len(parallel_rows)} rows)')

    # ── Convergence plot — all T values combined for full picture ─────────────
    # orig_rows: T=200,500,1000 for existing targets (from simulate.py, read-only)
    # parallel_rows: T=5000,10000 for all targets
    # new targets (sin_5pi/6pi/7pi) only appear in parallel_rows — that's fine
    combined = {(r['target'], r['m'], r['T']): r for r in orig_rows}
    for r in parallel_rows:
        combined[(r['target'], r['m'], r['T'])] = r
    combined = sorted(combined.values(), key=lambda r: (r['target'], r['m'], r['T']))

    if combined:
        make_convergence_plot(combined, out_path=CONV_PLOT)

    wall = time.time() - t_start
    print(f'\nDone.  {completed} new runs,  {skipped} skipped,  '
          f'wall time {wall/60:.1f} min')
