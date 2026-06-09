"""
simulate.py
===========
Numerically investigates Open Problem 4.1 (bias collapse in shallow ReLU networks).

Conjecture (slide 95):
    lim_{m -> inf} C(m, f*) = #{x in [-1,1] : f*''(x) = 0 and changes sign}

i.e. the number of bias clusters converges to the inflection-point count k of the
target as m grows, independently of how large m gets once past a threshold.

Goals per run
-------------
Goal 1 : count C(m, f*) for many (m, f*) pairs; build evidence that C -> k as m -> inf
Goal 3 : at convergence, verify ODE velocities ~ 0 and integrated residuals R_j ~ 0
         for every active neuron; count how many active neurons remain (should be <= k)

Outputs (per run, saved to figures/Replication data/{target}/m={m}/T={T}/)
--------------------------------------------------------------------------
  slide93_reproduction.png   -- bias trajectories, final fit, loss curve
  clusters_vs_inflections.png -- cluster locations vs analytically known inflection pts
  ode_verification.png        -- |da/dt|, |db/dt|, R_j at convergence  (Goal 3)
  convergence_check.csv       -- per-neuron da, db, R_j, active flag   (Goal 3)
  run_meta.csv                -- single-row summary metrics for this run

Summary outputs (written once after all runs)
---------------------------------------------
  figures/Replication data/run_summary.csv     -- one row per completed run
  figures/Replication data/convergence_plot.png -- C(m) vs m for every target
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display needed; runs headlessly overnight
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time

# ── Quadrature grid (fixed) ───────────────────────────────────────────────────
N_QUAD = 400
X_QUAD = np.linspace(-1.0, 1.0, N_QUAD)
DX     = X_QUAD[1] - X_QUAD[0]

# =============================================================================
# Core math
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def network(x, a, b):
    """f(x) = sum_j a_j * relu(x - b_j).  x:(N,), a,b:(m,) -> (N,)"""
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def make_ode(m, f_star):
    """Return the ODE RHS compatible with scipy solve_ivp.
    State vector y = [a_1,...,a_m, b_1,...,b_m], shape (2m,).
    """
    fstar_vals = f_star(X_QUAD)

    def ode(t, y):
        a, b      = y[:m], y[m:]
        residual  = network(X_QUAD, a, b) - fstar_vals        # (N,)
        relu_mat  = relu(X_QUAD[:, None] - b[None, :])        # (N, m)
        da        = -(residual[:, None] * relu_mat).sum(0) * DX
        cum_right = np.cumsum(residual[::-1])[::-1] * DX      # (N,)
        idx       = np.searchsorted(X_QUAD, b).clip(0, N_QUAD - 1)
        db        = a * cum_right[idx]
        return np.concatenate([da, db])

    return ode

def compute_ode_velocities(a, b, f_star):
    """Evaluate (da/dt, db/dt, R) at the given (a,b) state.
    R[j] = integral of residual from b[j] to 1  (bias ODE integrand before * a_j).
    """
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
    sorted_b = np.sort(biases)
    return 1 + int(np.sum(np.diff(sorted_b) > tol))

def get_cluster_centers(biases, tol=0.02):
    sorted_b = np.sort(biases)
    centers, group = [], [sorted_b[0]]
    for i in range(1, len(sorted_b)):
        if sorted_b[i] - sorted_b[i - 1] > tol:
            centers.append(float(np.mean(group)))
            group = []
        group.append(sorted_b[i])
    centers.append(float(np.mean(group)))
    return np.array(centers)

# =============================================================================
# Target functions
# Each entry: key -> (display label, analytical k, function)
#
# Analytical inflection-point counts (sign changes of f'' in (-1,1)):
#   sin(n*pi*x) : f'' = -n^2*pi^2 * sin(n*pi*x)
#                 zeros at x = j/n  for j in {-(n-1),...,-1,0,1,...,n-1}
#                 => 2n-1 inflection points
#   x^3         : f'' = 6x, one zero at x=0 => k=1
#   x^5 - 3x^3  : f'' = 20x^3 - 18x = 2x(10x^2 - 9)
#                 zeros: x=0, x=+/-sqrt(0.9) ~ +/-0.949  => k=3
# =============================================================================
TARGETS = {
    # k=1 — one inflection point
    'sin_1pi'  : (r'$\sin(\pi x)$',    1, lambda x: np.sin(np.pi * x)),
    'x_cubed'  : (r'$x^3$',            1, lambda x: x**3),

    # k=3 — three inflection points
    'sin_2pi'  : (r'$\sin(2\pi x)$',   3, lambda x: np.sin(2 * np.pi * x)),
    'poly_k3'  : (r'$x^5 - 3x^3$',    3, lambda x: x**5 - 3 * x**3),

    # k=5 — five inflection points
    'sin_3pi'  : (r'$\sin(3\pi x)$',   5, lambda x: np.sin(3 * np.pi * x)),

    # k=7 — seven inflection points
    'sin_4pi'  : (r'$\sin(4\pi x)$',   7, lambda x: np.sin(4 * np.pi * x)),
}

# Inflection-point x-locations (for plotting vertical lines)
# Used only in clusters_vs_inflections figure.
def get_inflection_locations(target_key):
    if target_key == 'sin_1pi':
        return np.array([0.0])
    elif target_key == 'x_cubed':
        return np.array([0.0])
    elif target_key == 'sin_2pi':
        return np.array([-0.5, 0.0, 0.5])
    elif target_key == 'poly_k3':
        return np.array([-np.sqrt(0.9), 0.0, np.sqrt(0.9)])
    elif target_key == 'sin_3pi':
        return np.array([-2/3, -1/3, 0.0, 1/3, 2/3])
    elif target_key == 'sin_4pi':
        return np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    return np.array([])

# =============================================================================
# Sweep parameters
# =============================================================================
M_VALUES = [50, 100, 250, 500, 1000, 1500]   # wide range to show C(m) -> k plateau
T_VALUES = [200, 500, 1000]                   # longer T for convergence evidence
N_SAVE   = 300     # trajectory snapshots (affects plot smoothness only)
SEED     = 42
ACTIVE_THRESHOLD = 0.05   # |a_j| > threshold * max|a_j| => active

FIG_BASE    = os.path.join('figures', 'Replication data')
SUMMARY_CSV = os.path.join(FIG_BASE, 'run_summary.csv')

# =============================================================================
# Single-run function
# =============================================================================

def run_one(target_key, target_label, f_star, k_true, m, T):
    """Simulate and save all outputs for one (target, m, T) combination.
    Returns a dict of key metrics, or None if the run was skipped.
    """
    out_dir  = os.path.join(FIG_BASE, target_key, f'm={m}', f'T={T}')
    f_traj   = os.path.join(out_dir, 'slide93_reproduction.png')
    f_clust  = os.path.join(out_dir, 'clusters_vs_inflections.png')
    f_ode    = os.path.join(out_dir, 'ode_verification.png')
    f_csv    = os.path.join(out_dir, 'convergence_check.csv')
    f_meta   = os.path.join(out_dir, 'run_meta.csv')

    if all(os.path.exists(p) for p in [f_traj, f_clust, f_ode, f_csv]):
        # Run is complete — read metrics from run_meta.csv if it exists
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
        return None   # completed before run_meta.csv existed; skip silently

    os.makedirs(out_dir, exist_ok=True)

    # ── Simulate ─────────────────────────────────────────────────────────────
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

    # ── Loss over time (sparse: every 5th snapshot) ───────────────────────────
    fstar_vals = f_star(X_QUAD)
    snap_idx   = np.arange(0, N_SAVE, 5)
    losses_sparse = np.array([
        0.5 * np.trapezoid(
            (network(X_QUAD, sol.y[:m, i], sol.y[m:, i]) - fstar_vals)**2, X_QUAD)
        for i in snap_idx
    ])
    t_sparse = sol.t[snap_idx]
    final_loss = float(0.5 * np.trapezoid(
        (network(X_QUAD, sol.y[:m, -1], sol.y[m:, -1]) - fstar_vals)**2, X_QUAD))

    # ── Final state ───────────────────────────────────────────────────────────
    a_final         = sol.y[:m, -1]
    b_final         = sol.y[m:, -1]
    x_plot          = np.linspace(-1.0, 1.0, 500)
    f_final         = network(x_plot, a_final, b_final)
    n_clusters      = count_clusters(b_final)
    cluster_centers = get_cluster_centers(b_final)

    # ── Goal 3: stationarity at convergence ───────────────────────────────────
    da_final, db_final, R_final = compute_ode_velocities(a_final, b_final, f_star)
    a_max     = max(float(np.abs(a_final).max()), 1e-12)
    is_active = np.abs(a_final) > ACTIVE_THRESHOLD * a_max
    n_active  = int(is_active.sum())

    # Sort by bias location for figures and CSV
    sort_idx = np.argsort(b_final)
    b_s      = b_final[sort_idx];   a_s = a_final[sort_idx]
    da_s     = da_final[sort_idx];  db_s = db_final[sort_idx]
    R_s      = R_final[sort_idx];   active_s = is_active[sort_idx]

    infl_x = get_inflection_locations(target_key)

    # ── Figure 1: trajectories / fit / loss  (Goal 1) ────────────────────────
    if not os.path.exists(f_traj):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'target={target_label},  m={m},  T={T}'
            f'  |  clusters={n_clusters},  k={k_true}',
            fontsize=12)

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
        ax.set_title(f'Loss (log scale)   final={final_loss:.2e}')
        ax.set_xlim([0, T]); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f_traj, bbox_inches='tight')
        plt.close()

    # ── Figure 2: cluster locations vs inflection points  (Goal 1) ───────────
    if not os.path.exists(f_clust):
        x_fine = np.linspace(-1.0, 1.0, 2000)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(
            f'target={target_label},  m={m},  T={T}  '
            f'|  clusters={n_clusters},  k={k_true}  '
            f'(C -> k: {n_clusters == k_true})',
            fontsize=11)
        ax.plot(x_fine, f_star(x_fine), 'k-', lw=2, label=f'Target {target_label}')
        ax.plot(x_plot, f_final, 'r-', lw=1.5, label='Final network')
        for cx in cluster_centers:
            ax.axvline(cx, color='steelblue', alpha=0.6, lw=0.8)
        for ix in infl_x:
            ax.axvline(ix, color='green', alpha=0.7, lw=1.2, linestyle='--')
        from matplotlib.lines import Line2D
        h, _ = ax.get_legend_handles_labels()
        h += [
            Line2D([0],[0], color='steelblue', lw=1.5, label=f'Bias clusters ({n_clusters})'),
            Line2D([0],[0], color='green', lw=1.5, ls='--', label=f'Inflection pts k={k_true}'),
        ]
        ax.legend(handles=h, fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_title('Cluster Locations vs Inflection Points')
        plt.tight_layout()
        plt.savefig(f_clust, bbox_inches='tight')
        plt.close()

    # ── Figure 3: ODE stationarity check  (Goal 3) ───────────────────────────
    if not os.path.exists(f_ode):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Stationarity check — target={target_label},  m={m},  T={T}\n'
            f'active={n_active},  k={k_true},  '
            f'active <= k: {n_active <= k_true}',
            fontsize=10)

        ax = axes[0]
        ax.scatter(b_s, np.abs(da_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=12)
        ax.set_xlabel('Bias $b_j$'); ax.set_ylabel('$|\\dot{a}_j|$')
        ax.set_title('Amplitude velocity at $t=T$\n(should be $\\approx 0$)')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(b_s, np.abs(db_s) + 1e-15, c=np.abs(a_s), cmap='viridis', s=12)
        ax.set_xlabel('Bias $b_j$'); ax.set_ylabel('$|\\dot{b}_j|$')
        ax.set_title('Bias velocity at $t=T$\n(should be $\\approx 0$)')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        ax = axes[2]
        sizes = np.clip(np.abs(a_s) / a_max * 80, 4, 80)
        sc = ax.scatter(b_s, R_s, c=np.abs(a_s), cmap='viridis', s=sizes)
        ax.scatter(b_s[active_s], R_s[active_s],
                   edgecolors='red', facecolors='none', s=70, lw=1.3,
                   label=f'Active ({n_active}), $R_j \\approx 0$ expected')
        ax.axhline(0.0, color='k', lw=1.0, linestyle='--')
        plt.colorbar(sc, ax=ax, label='$|a_j|$')
        ax.set_xlabel('Bias $b_j$')
        ax.set_ylabel('$R_j = \\int_{b_j}^{1}(f-f^*)\\,dx$')
        ax.set_title(f'Integrated residual\nactive={n_active} <= k={k_true}: {n_active <= k_true}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f_ode, bbox_inches='tight')
        plt.close()

    # ── CSV: per-neuron stationarity data  (Goal 3) ──────────────────────────
    if not os.path.exists(f_csv):
        R_tol = 1e-3
        with open(f_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'neuron_idx', 'b_j', 'a_j',
                'da_dt', 'db_dt', 'R_j',
                'is_active', 'R_near_zero',
            ])
            for i in range(m):
                writer.writerow([
                    sort_idx[i],
                    f'{b_s[i]:.6f}',  f'{a_s[i]:.6f}',
                    f'{da_s[i]:.6e}', f'{db_s[i]:.6e}', f'{R_s[i]:.6e}',
                    int(active_s[i]), int(abs(R_s[i]) < R_tol),
                ])

    result = {
        'target': target_key, 'm': m, 'T': T,
        'k_true': k_true, 'n_clusters': n_clusters,
        'n_active': n_active, 'loss': final_loss,
        'max_da': float(np.abs(da_final).max()),
        'max_db': float(np.abs(db_final).max()),
        'active_leq_k': int(n_active <= k_true),
    }

    # ── run_meta.csv: single-row summary so skipped runs can be recovered ─────
    meta_fields = ['target','m','T','k_true','n_clusters','n_active',
                   'loss','max_da','max_db','active_leq_k']
    with open(f_meta, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerow({k: result[k] for k in meta_fields})

    return result

# =============================================================================
# Summary: convergence plot  C(m) vs m  for each target
# =============================================================================

def make_convergence_plot(summary_rows):
    """Plot C(m) vs m for every target and every T value.
    One figure per target, saved alongside run_summary.csv.
    """
    from collections import defaultdict
    import itertools

    targets_present = list(dict.fromkeys(r['target'] for r in summary_rows))
    T_present       = sorted(set(r['T'] for r in summary_rows))
    markers         = ['o', 's', '^', 'D', 'v', 'P']
    colors          = plt.cm.tab10(np.linspace(0, 0.6, len(T_present)))

    n_targets = len(targets_present)
    fig, axes = plt.subplots(
        2, -(-n_targets // 2),    # ceiling division
        figsize=(5 * -(-n_targets // 2), 8),
        squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, tkey in enumerate(targets_present):
        ax = axes_flat[ax_i]
        rows_t = [r for r in summary_rows if r['target'] == tkey]
        k_true = rows_t[0]['k_true']
        label  = list(TARGETS[tkey][0:1])[0] if tkey in TARGETS else tkey

        for ti, T in enumerate(T_present):
            rows_mT = sorted([r for r in rows_t if r['T'] == T], key=lambda r: r['m'])
            if not rows_mT:
                continue
            ms = [r['m'] for r in rows_mT]
            cs = [r['n_clusters'] for r in rows_mT]
            ax.plot(ms, cs, marker=markers[ti % len(markers)],
                    color=colors[ti], lw=1.5, ms=6, label=f'T={T}')

        # Horizontal line at the analytical k
        ax.axhline(k_true, color='crimson', lw=2, linestyle='--',
                   label=f'k={k_true} (inflection pts)')
        ax.set_xlabel('Width $m$')
        ax.set_ylabel('Cluster count $C(m, f^*)$')
        ax.set_title(f'{TARGETS[tkey][0] if tkey in TARGETS else tkey}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for ax_i in range(len(targets_present), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(
        'Open Problem 4.1: $C(m,f^*) \\to k$ as $m \\to \\infty$\n'
        'Dashed line = analytical inflection-point count $k$',
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_BASE, 'convergence_plot.png')
    plt.savefig(out, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Convergence plot saved to {out}')

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(FIG_BASE, exist_ok=True)

    total     = len(TARGETS) * len(M_VALUES) * len(T_VALUES)
    done      = 0
    skipped   = 0
    new_rows  = []

    # Load any previously written summary rows so the convergence plot is cumulative
    existing_rows = []
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append({
                    'target': row['target'], 'm': int(row['m']), 'T': int(row['T']),
                    'k_true': int(row['k_true']), 'n_clusters': int(row['n_clusters']),
                    'n_active': int(row['n_active']), 'loss': float(row['loss']),
                    'max_da': float(row['max_da']), 'max_db': float(row['max_db']),
                    'active_leq_k': int(row['active_leq_k']),
                })

    existing_keys = {(r['target'], r['m'], r['T']) for r in existing_rows}

    print(f"{'='*70}")
    print(f"Open Problem 4.1 — Bias Collapse Replication")
    print(f"Targets  : {list(TARGETS.keys())}")
    print(f"m values : {M_VALUES}")
    print(f"T values : {T_VALUES}")
    print(f"Total    : {total} runs  ({len(existing_keys)} already in summary)")
    print(f"{'='*70}\n")

    for target_key, (target_label, k_true, f_star) in TARGETS.items():
        for m in M_VALUES:
            for T in T_VALUES:
                done += 1
                tag = f'[{done:>3}/{total}]  {target_key:<12}  m={m:<5}  T={T:<5}'

                result = run_one(target_key, target_label, f_star, k_true, m, T)

                if result is None:
                    skipped += 1
                    print(f'{tag}  SKIP (no run_meta.csv; excluded from summary)')
                elif result.pop('_skipped', False):
                    skipped += 1
                    new_rows.append(result)
                    print(f'{tag}  SKIP (recovered from run_meta.csv) '
                          f'clusters={result["n_clusters"]}  k={k_true}')
                else:
                    new_rows.append(result)
                    print(
                        f'{tag}  '
                        f'clusters={result["n_clusters"]:<4}  '
                        f'k={k_true}  '
                        f'C->k: {result["n_clusters"]==k_true}  '
                        f'active={result["n_active"]:<4}  '
                        f'active<=k: {bool(result["active_leq_k"])}  '
                        f'loss={result["loss"]:.4e}  '
                        f'max|da|={result["max_da"]:.2e}  '
                        f'max|db|={result["max_db"]:.2e}'
                    )

    # ── Write / append summary CSV ────────────────────────────────────────────
    all_rows = existing_rows.copy()
    new_keys = {(r['target'], r['m'], r['T']) for r in new_rows}
    for r in new_rows:
        key = (r['target'], r['m'], r['T'])
        # Update if already present, otherwise append
        all_rows = [x for x in all_rows if (x['target'], x['m'], x['T']) != key]
        all_rows.append(r)

    fieldnames = ['target','m','T','k_true','n_clusters','n_active',
                  'loss','max_da','max_db','active_leq_k']
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_rows, key=lambda x: (x['target'], x['m'], x['T'])):
            writer.writerow({k: r[k] for k in fieldnames})
    print(f'\nRun summary written to {SUMMARY_CSV}')

    # ── Convergence plot ──────────────────────────────────────────────────────
    if all_rows:
        make_convergence_plot(all_rows)

    print(f'\nDone. {done - skipped} new runs, {skipped} skipped.')
