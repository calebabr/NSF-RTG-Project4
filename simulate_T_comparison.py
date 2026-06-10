"""
simulate_T_comparison.py
========================
Runs the same ODE gradient-flow simulation as simulate_parallel.py but at
T=250 and T=750, for a targeted subset of (target, m) combinations.

Purpose
-------
Compare C(m, T=250) vs C(m, T=500) vs C(m, T=750) to determine whether
cluster count convergence is finite-time (C locks in early and stays) or
asymptotic (C still drifting at T=500).

Targeted runs
-------------
  - Non-stationary at T=500 (max_da >= 0.01): the runs most likely to change
  - C=k hits at T=500: verify they were already stable at T=250
  - Below-k stationary cases: check whether longer T changes anything
  - Dagger cases: see if cluster structure emerges or dissolves further

Outputs
-------
  Per run : figures/Replication data/{target}/m={m}/T={T}/
                (same format as simulate_parallel.py)
  Summary : figures/Replication data/T_comparison_summary.csv
  Plot    : figures/Replication data/T_comparison_plot.png
              Shows C and max_da at T=250, T=500, T=750 for each targeted run.

Usage
-----
    python simulate_T_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Quadrature grid  (identical to simulate_parallel.py)
# =============================================================================
N_QUAD = 200
X_QUAD = np.linspace(-1.0, 1.0, N_QUAD)
DX     = X_QUAD[1] - X_QUAD[0]

# =============================================================================
# Core math  (identical to simulate_parallel.py)
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
# Target functions  (named defs for multiprocessing pickle)
# =============================================================================

def f_sin_1pi(x): return np.sin(    np.pi * x)
def f_x_cubed(x): return x**3
def f_sin_2pi(x): return np.sin(2 * np.pi * x)
def f_poly_k3(x): return x**5 - 3 * x**3
def f_sin_3pi(x): return np.sin(3 * np.pi * x)
def f_sin_4pi(x): return np.sin(4 * np.pi * x)
def f_sin_5pi(x): return np.sin(5 * np.pi * x)
def f_sin_6pi(x): return np.sin(6 * np.pi * x)
def f_sin_7pi(x): return np.sin(7 * np.pi * x)

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
# Configuration
# =============================================================================

T_VALUES = [250, 750]   # T=500 already exists; we add 250 and 750

# Targeted (target, m) pairs chosen for the T comparison:
#   - Non-stationary at T=500 (max_da >= 0.01)
#   - C=k hits at T=500 (to confirm stability)
#   - Below-k stationary cases (to check if longer T changes C)
#   - Key dagger cases (adaptive threshold recovered C=k)
TARGETED_RUNS = [
    # Non-stationary at T=500
    ('sin_4pi',  500),
    ('sin_4pi', 1000),
    ('sin_5pi', 1000),
    ('sin_5pi', 5000),
    ('sin_6pi', 1000),
    ('sin_6pi', 3500),
    ('sin_7pi',  250),
    ('sin_7pi', 2000),
    ('sin_7pi', 5000),
    # C=k hits at T=500
    ('sin_1pi', 1000),
    ('sin_1pi', 2000),
    ('sin_1pi', 5000),
    ('sin_3pi', 1000),
    ('sin_4pi', 1500),
    # Below-k stationary cases
    ('sin_2pi', 1000),
    ('sin_4pi', 2000),
    ('poly_k3', 5000),
    # Dagger cases where adaptive threshold found C=k
    ('sin_3pi', 3500),
    ('sin_4pi', 5000),
    # Additional trajectory context
    ('sin_7pi', 1000),
    ('sin_6pi', 2000),
    ('sin_5pi', 2000),
]

SEED             = 42          # must match simulate_parallel.py
N_SAVE           = 300
ACTIVE_THRESHOLD = 0.05
FIG_BASE         = os.path.join('figures', 'Replication data')
SUMMARY_CSV      = os.path.join(FIG_BASE, 'T_comparison_summary.csv')
COMP_PLOT        = os.path.join(FIG_BASE, 'T_comparison_plot.png')
META_FIELDS      = ['target', 'm', 'T', 'k_true', 'n_clusters', 'n_active',
                    'loss', 'max_da', 'max_db', 'active_leq_k']

# =============================================================================
# Single-run worker
# =============================================================================

def run_one(args):
    target_key, m, T = args
    target_label, k_true, f_star = TARGETS[target_key]

    out_dir = os.path.join(FIG_BASE, target_key, f'm={m}', f'T={T}')
    f_traj  = os.path.join(out_dir, 'slide93_reproduction.png')
    f_clust = os.path.join(out_dir, 'clusters_vs_inflections.png')
    f_ode   = os.path.join(out_dir, 'ode_verification.png')
    f_csv   = os.path.join(out_dir, 'convergence_check.csv')
    f_meta  = os.path.join(out_dir, 'run_meta.csv')

    # Skip if already complete
    if all(os.path.exists(p) for p in [f_traj, f_clust, f_ode, f_csv, f_meta]):
        with open(f_meta, newline='') as mf:
            row = next(csv.DictReader(mf))
        return {
            'target': target_key, 'm': m, 'T': T,
            'k_true':       int(row['k_true']),
            'n_clusters':   int(row['n_clusters']),
            'n_active':     int(row['n_active']),
            'loss':         float(row['loss']),
            'max_da':       float(row['max_da']),
            'max_db':       float(row['max_db']),
            'active_leq_k': int(row['active_leq_k']),
            '_skipped': True,
        }

    os.makedirs(out_dir, exist_ok=True)

    # Simulate
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

    # Loss trajectory
    fstar_vals    = f_star(X_QUAD)
    snap_idx      = np.arange(0, N_SAVE, 5)
    losses_sparse = np.array([
        0.5 * np.trapz(
            (network(X_QUAD, sol.y[:m, i], sol.y[m:, i]) - fstar_vals)**2, X_QUAD)
        for i in snap_idx
    ])
    t_sparse   = sol.t[snap_idx]
    final_loss = float(0.5 * np.trapz(
        (network(X_QUAD, sol.y[:m, -1], sol.y[m:, -1]) - fstar_vals)**2, X_QUAD))

    # Final state
    a_final         = sol.y[:m, -1]
    b_final         = sol.y[m:, -1]
    x_plot          = np.linspace(-1.0, 1.0, 500)
    f_final         = network(x_plot, a_final, b_final)
    n_clusters      = count_clusters(b_final)
    cluster_centers = get_cluster_centers(b_final)

    da_final, db_final, R_final = compute_ode_velocities(a_final, b_final, f_star)
    a_max     = max(float(np.abs(a_final).max()), 1e-12)
    is_active = np.abs(a_final) > ACTIVE_THRESHOLD * a_max
    n_active  = int(is_active.sum())

    sort_idx = np.argsort(b_final)
    b_s  = b_final[sort_idx];  a_s  = a_final[sort_idx]
    da_s = da_final[sort_idx]; db_s = db_final[sort_idx]
    R_s  = R_final[sort_idx];  active_s = is_active[sort_idx]
    infl_x = get_inflection_locations(target_key)

    # Figure 1: trajectories / fit / loss
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
    y0, y1 = ax.get_ylim()
    tick_h = (y1 - y0) * 0.06
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

    # Figure 2: clusters vs inflection points
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

    # Figure 3: ODE stationarity check
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'Stationarity — target={target_label},  m={m},  T={T}\n'
        f'active={n_active},  k={k_true},  active<=k: {n_active<=k_true}',
        fontsize=10)

    ax = axes[0]
    ax.scatter(b_s, np.abs(da_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
    ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{a}_j|$')
    ax.set_title('Amplitude velocity')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(b_s, np.abs(db_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
    ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{b}_j|$')
    ax.set_title('Bias velocity')
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
    ax.set_ylabel('$R_j$')
    ax.set_title(f'Integrated residual  active={n_active} <= k={k_true}: {n_active<=k_true}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f_ode, bbox_inches='tight')
    plt.close()

    # convergence_check.csv
    R_tol = 1e-3
    with open(f_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['neuron_idx', 'b_j', 'a_j',
                         'da_dt', 'db_dt', 'R_j', 'is_active', 'R_near_zero'])
        for i in range(m):
            writer.writerow([
                sort_idx[i],
                f'{b_s[i]:.6f}',  f'{a_s[i]:.6f}',
                f'{da_s[i]:.6e}', f'{db_s[i]:.6e}', f'{R_s[i]:.6e}',
                int(active_s[i]), int(abs(R_s[i]) < R_tol),
            ])

    # run_meta.csv
    result = {
        'target': target_key, 'm': m, 'T': T,
        'k_true':       k_true,
        'n_clusters':   n_clusters,
        'n_active':     n_active,
        'loss':         final_loss,
        'max_da':       float(np.abs(da_final).max()),
        'max_db':       float(np.abs(db_final).max()),
        'active_leq_k': int(n_active <= k_true),
    }
    with open(f_meta, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerow({k: result[k] for k in META_FIELDS})

    result['_elapsed'] = round(elapsed, 1)
    return result

# =============================================================================
# Comparison plot
# =============================================================================

def make_comparison_plot(all_rows):
    """
    For each targeted (target, m) pair, plot C and max_da across T=250, 500, 750.
    Rows at T=500 are loaded from existing run_summary_parallel.csv.
    """
    T_COLORS = {250: '#2ca02c', 500: '#1f77b4', 750: '#d62728'}
    T_MARKERS = {250: '^', 500: 'o', 750: 's'}

    # Load T=500 data from existing summary
    t500_data = {}
    existing_csv = os.path.join(FIG_BASE, 'run_summary_parallel.csv')
    if os.path.exists(existing_csv):
        with open(existing_csv, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['target'], int(row['m']))
                t500_data[key] = {
                    'n_clusters': int(row['n_clusters']),
                    'max_da': float(row['max_da']),
                    'loss': float(row['loss']),
                    'k_true': int(row['k_true']),
                }

    # Index new rows by (target, m, T)
    new_data = {}
    for r in all_rows:
        new_data[(r['target'], r['m'], r['T'])] = r

    # Build per-run comparison table
    n_runs = len(TARGETED_RUNS)
    ncols  = 4
    nrows  = -(-n_runs // ncols)

    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(ncols * 6, nrows * 3.5))
    axes = axes.reshape(nrows, ncols * 2)

    for idx, (target, m) in enumerate(TARGETED_RUNS):
        row_i = idx // ncols
        col_i = (idx % ncols) * 2

        ax_c  = axes[row_i, col_i]      # cluster count
        ax_da = axes[row_i, col_i + 1]  # max_da

        k_true = TARGETS[target][1]
        label  = TARGETS[target][0]

        for T in [250, 500, 750]:
            if T == 500:
                entry = t500_data.get((target, m))
            else:
                entry = new_data.get((target, m, T))
            if entry is None:
                continue
            c   = entry['n_clusters']
            da  = entry['max_da']
            col = T_COLORS[T]
            mrk = T_MARKERS[T]
            ax_c.scatter([T], [c],  color=col, marker=mrk, s=80, zorder=5)
            ax_da.scatter([T], [da], color=col, marker=mrk, s=80, zorder=5)

        # Connect points
        pts_c, pts_da = [], []
        for T in [250, 500, 750]:
            entry = t500_data.get((target, m)) if T == 500 else new_data.get((target, m, T))
            if entry:
                pts_c.append((T, entry['n_clusters']))
                pts_da.append((T, entry['max_da']))
        if len(pts_c) > 1:
            ax_c.plot([p[0] for p in pts_c], [p[1] for p in pts_c],
                      'k-', lw=1, alpha=0.5, zorder=4)
        if len(pts_da) > 1:
            ax_da.plot([p[0] for p in pts_da], [p[1] for p in pts_da],
                       'k-', lw=1, alpha=0.5, zorder=4)

        ax_c.axhline(k_true, color='red', lw=1.2, ls=':', alpha=0.7)
        ax_da.axhline(0.01, color='orange', lw=1, ls='--', alpha=0.7,
                      label='stationarity threshold')

        ax_c.set_title(f'{label}\nm={m}', fontsize=8)
        ax_c.set_ylabel('C', fontsize=8)
        ax_c.set_xticks([250, 500, 750])
        ax_c.tick_params(labelsize=7)
        ax_c.grid(True, alpha=0.3)

        ax_da.set_title(f'max|da/dt|\nm={m}', fontsize=8)
        ax_da.set_ylabel('max|da/dt|', fontsize=8)
        ax_da.set_xticks([250, 500, 750])
        ax_da.tick_params(labelsize=7)
        ax_da.set_yscale('log')
        ax_da.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_runs, nrows * ncols):
        row_i = idx // ncols
        col_i = (idx % ncols) * 2
        axes[row_i, col_i].set_visible(False)
        axes[row_i, col_i + 1].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker=T_MARKERS[T], color=T_COLORS[T],
               lw=0, ms=8, label=f'T={T}')
        for T in [250, 500, 750]
    ] + [
        Line2D([0],[0], color='red', lw=1.2, ls=':', label='k (true)'),
        Line2D([0],[0], color='orange', lw=1, ls='--', label='stationarity 0.01'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        'T comparison: C(m) and max|da/dt| at T=250, 500, 750\n'
        'Red dotted = k.  Orange dashed = stationarity threshold (0.01).',
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(COMP_PLOT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison plot -> {COMP_PLOT}')

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(FIG_BASE, exist_ok=True)

    # sin_7pi T=750 is redundant: a T=1000 run is already in progress.
    SKIP = {('sin_7pi', T) for T in [750]}

    jobs = sorted(
        [(t, m, T) for (t, m) in TARGETED_RUNS for T in T_VALUES
         if (t, T) not in SKIP],
        key=lambda x: (x[2], x[1], x[0])  # sort by T, then m, then target
    )

    n_workers = max(1, cpu_count() - 2)

    print('=' * 72)
    print('simulate_T_comparison.py')
    print(f'T values     : {T_VALUES}')
    print(f'Targeted runs: {len(TARGETED_RUNS)} (target, m) pairs')
    print(f'Total jobs   : {len(jobs)}')
    print(f'Workers      : {n_workers}')
    print('=' * 72)

    t_start   = time.time()
    all_rows  = []
    skipped   = 0
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            if result is None:
                continue
            was_skipped = result.pop('_skipped', False)
            elapsed     = result.pop('_elapsed', None)
            all_rows.append(result)
            tag = (f'{result["target"]:<12}  m={result["m"]:<5}  T={result["T"]:<4}  '
                   f'C={result["n_clusters"]:<4}  k={result["k_true"]}  '
                   f'C=k: {result["n_clusters"] == result["k_true"]}  '
                   f'max_da={result["max_da"]:.3e}  loss={result["loss"]:.3e}')
            if was_skipped:
                skipped += 1
                print(f'  SKIP  {tag}')
            else:
                completed += 1
                t_str = f'  [{elapsed:.0f}s]' if elapsed else ''
                print(f'  DONE{t_str}  {tag}')

    # Write summary CSV
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for r in sorted(all_rows, key=lambda r: (r['target'], r['m'], r['T'])):
            writer.writerow({k: r[k] for k in META_FIELDS})
    print(f'\nSummary -> {SUMMARY_CSV}  ({len(all_rows)} rows)')

    # Print C comparison table
    print('\n--- C comparison: T=250 vs T=500 vs T=750 ---')
    t500 = {}
    existing = os.path.join(FIG_BASE, 'run_summary_parallel.csv')
    if os.path.exists(existing):
        with open(existing, newline='') as f:
            for row in csv.DictReader(f):
                t500[(row['target'], int(row['m']))] = int(row['n_clusters'])

    indexed = {(r['target'], r['m'], r['T']): r for r in all_rows}
    print(f'{"Target":<12} {"m":>6}  {"k":>3}  {"T=250":>6}  {"T=500":>6}  {"T=750":>6}  {"stable?":>8}')
    print('-' * 60)
    for target, m in TARGETED_RUNS:
        k = TARGETS[target][1]
        c250 = indexed.get((target, m, 250), {}).get('n_clusters', '?')
        c500 = t500.get((target, m), '?')
        c750 = indexed.get((target, m, 750), {}).get('n_clusters', '?')
        stable = '✓' if c250 == c500 == c750 else ''
        print(f'{target:<12} {m:>6}  {k:>3}  {str(c250):>6}  {str(c500):>6}  {str(c750):>6}  {stable:>8}')

    make_comparison_plot(all_rows)

    wall = time.time() - t_start
    print(f'\nDone.  {completed} new,  {skipped} skipped,  wall time {wall/60:.1f} min')
