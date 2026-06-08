"""
simulate_parallel.py
====================
Parallel ODE gradient-flow simulation for Open Problem 4.1.

Conjecture:
    lim_{m -> inf} C(m, f*) = #{x in [-1,1] : f*''(x) = 0 and changes sign}

Design
------
  Fixed T = T_FINAL (single integration time — once the ODE has equilibrated,
  more T does not change the cluster count; only m matters for the conjecture).
  Flat m sweep across all targets — no Phase 2 logic, no convergence gating.
  Jobs are ordered (m, target) so every target has small-m data even if
  interrupted early.

Output
------
  Per run : figures/Replication data/{target}/m={m}/T={T_FINAL}/
                slide93_reproduction.png
                clusters_vs_inflections.png
                ode_verification.png
                convergence_check.csv
                run_meta.csv
  Summary : figures/Replication data/run_summary_parallel.csv
  Plot    : figures/Replication data/convergence_plot_parallel.png

Usage
-----
    # Run from the project root (MathProject4/) — outputs go to figures/Replication data/
    pip install -r requirements.txt
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
# Core math
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
# Target functions  (named defs — not lambdas — for multiprocessing pickle)
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
# Sweep parameters
# =============================================================================

T_FINAL  = 500                                        # fixed integration time
M_VALUES = [50, 100, 250, 500, 1000, 2000, 3500, 5000]  # flat m sweep

N_SAVE           = 300
SEED             = 42
ACTIVE_THRESHOLD = 0.05

FIG_BASE    = os.path.join('figures', 'Replication data')
SUMMARY_CSV = os.path.join(FIG_BASE, 'run_summary_parallel.csv')
CONV_PLOT   = os.path.join(FIG_BASE, 'convergence_plot_parallel.png')

META_FIELDS = ['target', 'm', 'T', 'k_true', 'n_clusters', 'n_active',
               'loss', 'max_da', 'max_db', 'active_leq_k']

# =============================================================================
# Single-run worker
# =============================================================================

def run_one(args):
    """
    Simulate one (target_key, m) combination at T=T_FINAL.
    Returns a metrics dict (with '_skipped' True if recovered from run_meta.csv).
    """
    target_key, m = args
    T = T_FINAL
    target_label, k_true, f_star = TARGETS[target_key]

    out_dir = os.path.join(FIG_BASE, target_key, f'm={m}', f'T={T}')
    f_traj  = os.path.join(out_dir, 'slide93_reproduction.png')
    f_clust = os.path.join(out_dir, 'clusters_vs_inflections.png')
    f_ode   = os.path.join(out_dir, 'ode_verification.png')
    f_csv   = os.path.join(out_dir, 'convergence_check.csv')
    f_meta  = os.path.join(out_dir, 'run_meta.csv')

    # ── Skip if already complete ──────────────────────────────────────────────
    if all(os.path.exists(p) for p in [f_traj, f_clust, f_ode, f_csv]):
        if os.path.exists(f_meta):
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
        return None   # pre-run_meta era, skip silently

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

    # ── Loss ──────────────────────────────────────────────────────────────────
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

    # ── Final state ───────────────────────────────────────────────────────────
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

    # ── Figure 1: trajectories / fit / loss ───────────────────────────────────
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

    # ── Figure 2: clusters vs inflection points ───────────────────────────────
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

    # ── Figure 3: ODE stationarity check ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'Stationarity — target={target_label},  m={m},  T={T}\n'
        f'active={n_active},  k={k_true},  active<=k: {n_active<=k_true}',
        fontsize=10)

    ax = axes[0]
    ax.scatter(b_s, np.abs(da_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
    ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{a}_j|$')
    ax.set_title('Amplitude velocity  ($\\approx 0$ at stationarity)')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(b_s, np.abs(db_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=10)
    ax.set_xlabel('$b_j$'); ax.set_ylabel('$|\\dot{b}_j|$')
    ax.set_title('Bias velocity  ($\\approx 0$ at stationarity)')
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

    # ── convergence_check.csv ─────────────────────────────────────────────────
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

    # ── run_meta.csv ──────────────────────────────────────────────────────────
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
# Convergence plot  (single series per target — T is fixed)
# =============================================================================

def make_convergence_plot(summary_rows, out_path=CONV_PLOT):
    targets_present = list(dict.fromkeys(
        k for k in TARGETS if any(r['target'] == k for r in summary_rows)
    ))
    n_t   = len(targets_present)
    ncols = 3
    nrows = -(-n_t // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, tkey in enumerate(targets_present):
        ax     = axes_flat[ax_i]
        k_true = TARGETS[tkey][1]
        pts    = sorted([r for r in summary_rows if r['target'] == tkey],
                        key=lambda r: r['m'])
        if pts:
            ms = [r['m']          for r in pts]
            cs = [r['n_clusters'] for r in pts]
            ax.plot(ms, cs, marker='o', color='steelblue',
                    lw=1.5, ms=5, label=f'T={T_FINAL}')
        ax.axhline(k_true, color='crimson', lw=2, linestyle='--',
                   label=f'k={k_true}')
        ax.set_xlabel('Width $m$')
        ax.set_ylabel('Cluster count $C(m,f^*)$')
        ax.set_title(TARGETS[tkey][0])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax_i in range(len(targets_present), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(
        f'Open Problem 4.1 — $C(m,f^*) \\to k$ as $m \\to \\infty$'
        f'  (ODE flow, T={T_FINAL})\n'
        f'Crimson dashed = analytical inflection-point count $k$'
        f'  |  {len(summary_rows)} completed runs',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Convergence plot -> {out_path}')

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(FIG_BASE, exist_ok=True)

    def load_csv(path):
        rows = []
        if os.path.exists(path):
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    rows.append({
                        'target':       row['target'],
                        'm':            int(row['m']),
                        'T':            int(row['T']),
                        'k_true':       int(row['k_true']),
                        'n_clusters':   int(row['n_clusters']),
                        'n_active':     int(row['n_active']),
                        'loss':         float(row['loss']),
                        'max_da':       float(row['max_da']),
                        'max_db':       float(row['max_db']),
                        'active_leq_k': int(row['active_leq_k']),
                    })
        return rows

    existing_rows = load_csv(SUMMARY_CSV)
    n_workers     = max(1, cpu_count() - 2)

    jobs = sorted(
        [(t, m) for t in TARGETS for m in M_VALUES],
        key=lambda x: (x[1], x[0])   # all targets at m=50 before m=100, etc.
    )

    print('=' * 72)
    print('simulate_parallel.py — Open Problem 4.1  (ODE gradient flow)')
    print(f'T_FINAL      : {T_FINAL}  (fixed — m is the sweep axis)')
    print(f'M_VALUES     : {M_VALUES}')
    print(f'Targets      : {list(TARGETS.keys())}')
    print(f'Total jobs   : {len(jobs)}  ({len(TARGETS)} targets × {len(M_VALUES)} m values)')
    print(f'Workers      : {n_workers}  (of {cpu_count()} logical CPUs)')
    print(f'Previous rows: {len(existing_rows)}')
    print(f'Summary  ->  {SUMMARY_CSV}')
    print(f'Plot     ->  {CONV_PLOT}')
    print('=' * 72)

    t_start   = time.time()
    all_new   = []
    skipped   = 0
    completed = 0

    print(f'\n{"=" * 72}')
    print('Running flat m sweep — all targets, all m values')
    print(f'{"=" * 72}')

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            if result is None:
                skipped += 1
                continue
            was_skipped = result.pop('_skipped', False)
            elapsed     = result.pop('_elapsed', None)
            all_new.append(result)
            tag = (f'{result["target"]:<12}  m={result["m"]:<5}  T={result["T"]:<5}  '
                   f'clusters={result["n_clusters"]:<4}  k={result["k_true"]}  '
                   f'C=k: {result["n_clusters"] == result["k_true"]}  '
                   f'loss={result["loss"]:.3e}  '
                   f'max|db|={result["max_db"]:.2e}')
            if was_skipped:
                skipped += 1
                print(f'  SKIP (meta)  {tag}')
            else:
                completed += 1
                t_str = f'  [{elapsed:.0f}s]' if elapsed else ''
                print(f'  DONE{t_str}  {tag}')

    # ── Write summary CSV ─────────────────────────────────────────────────────
    merged = {(r['target'], r['m']): r for r in existing_rows}
    for r in all_new:
        merged[(r['target'], r['m'])] = r
    final_rows = sorted(merged.values(), key=lambda r: (r['target'], r['m']))

    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for r in final_rows:
            writer.writerow({k: r[k] for k in META_FIELDS})
    print(f'\nSummary -> {SUMMARY_CSV}  ({len(final_rows)} rows)')

    if final_rows:
        make_convergence_plot(final_rows)

    wall = time.time() - t_start
    print(f'\nDone.  {completed} new runs,  {skipped} skipped,  '
          f'wall time {wall / 60:.1f} min')
