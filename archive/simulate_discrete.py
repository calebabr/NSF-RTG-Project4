"""
simulate_discrete.py
====================
Discrete gradient descent version of the bias collapse simulation.

Uses the same shallow ReLU network and target functions as simulate_parallel.py
but replaces the ODE solver (RK45) with explicit gradient descent updates —
the same approach used in the original slide code.

Learning rate scaling
---------------------
  LR_SCALE = False  →  constant lr = LR_BASE = 0.01 (slide code value).
                        Works for m <= 250 but diverges at m >= 500.
                        Output: figures/Discrete GD/

  LR_SCALE = True   →  lr = LR_BASE / sqrt(m / 50) so that lr = 0.01 at m=50
                        and decreases as 1/sqrt(m) for larger m.
                        Steps also scale: n_steps = min(N_STEPS_MAX,
                          int(N_STEPS_BASE * sqrt(m / 50))) to maintain
                        the same total gradient "work" across all m.
                        Output: figures/Discrete GD Scaled/

Network
-------
  f(x) = sum_j a_j * relu(x - b_j),   x in [-1, 1]

Two-phase execution
-------------------
  Phase 1 : base m sweep [50, 100, 250, 500, 1000] for all targets.
  Phase 2 : targets not converged at m=1000 extend through [1500, 2500, …, 5000].

Output
------
  Per run : {FIG_BASE}/{target}/m={m}/steps={n_steps}/
              bias_trajectories.png
              clusters_vs_inflections.png
              convergence_check.csv
              run_meta.csv
  Global  : {FIG_BASE}/run_summary_discrete.csv
            {FIG_BASE}/convergence_plot_discrete.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Hyperparameters
# =============================================================================

LR_SCALE     = True         # True  → lr scales as 1/sqrt(m/50), stable at all m
                             # False → constant lr = LR_BASE (slide code; diverges m>=500)
LR_BASE      = 1e-3         # lr at m=50 when LR_SCALE=True; constant lr when LR_SCALE=False
N_STEPS_BASE = 50_000       # steps at m=50 (LR_SCALE=True) or for all m (LR_SCALE=False)
N_STEPS_MAX  = 200_000      # cap on step count (LR_SCALE=True only)
LOG_EVERY    = 1_000        # log bias snapshot and loss every this many steps

N_QUAD           = 400
X_QUAD           = np.linspace(-1.0, 1.0, N_QUAD)
DX               = X_QUAD[1] - X_QUAD[0]
CLUSTER_TOL      = 0.02
ACTIVE_THRESHOLD = 0.05
SEED             = 42

# =============================================================================
# Sweep / convergence parameters  (mirrors simulate_parallel.py)
# =============================================================================

M_VALUES_ALL       = [50, 100, 250, 500, 1000]  # Phase 1 base sweep
PHASE2_M_FIXED     = [1500, 2500]               # Phase 2 always begins with these
PHASE2_M_INCREMENT = 500                        # step size after 2500
PHASE2_M_MAX       = 5000                       # hard cap
CONV_THRESHOLD     = 1                          # |C - k| <= this counts as converged

# =============================================================================
# Output paths  (depend on LR_SCALE)
# =============================================================================

FIG_BASE    = os.path.join('figures', 'Discrete GD')
SUMMARY_CSV = os.path.join(FIG_BASE, 'run_summary_discrete.csv')
CONV_PLOT   = os.path.join(FIG_BASE, 'convergence_plot_discrete.png')

META_FIELDS = ['target', 'm', 'steps', 'lr_eff', 'k_true', 'n_clusters',
               'n_active', 'loss', 'max_da_grad', 'max_db_grad', 'active_leq_k']

# =============================================================================
# Target functions
# (named defs — not lambdas — so multiprocessing can pickle on Windows)
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

INFLECTIONS = {
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

# =============================================================================
# Math helpers
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)


def count_clusters(biases, tol=CLUSTER_TOL):
    s = np.sort(biases)
    return 1 + int(np.sum(np.diff(s) > tol))


def get_cluster_centers(biases, tol=CLUSTER_TOL):
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
# Per-run lr / step count (called inside run_one so workers get same value)
# =============================================================================

def _lr_and_steps(m):
    """Return (lr, n_steps) for a given m according to LR_SCALE setting."""
    if LR_SCALE:
        scale    = np.sqrt(float(m) / 50.0)           # = 1 at m=50
        lr       = LR_BASE / scale                     # lr = 0.01 at m=50
        n_steps  = min(N_STEPS_MAX, int(N_STEPS_BASE * scale))
    else:
        lr      = LR_BASE
        n_steps = N_STEPS_BASE
    return lr, n_steps


# =============================================================================
# Single-run worker  (module-level so multiprocessing can pickle on Windows)
# =============================================================================

def run_one(args):
    """
    Train a shallow ReLU network via discrete GD on one (target_key, m) combo.
    Returns a metrics dict; includes '_skipped': True if read from run_meta.csv.
    """
    target_key, m = args
    target_label, k_true, f_star_fn = TARGETS[target_key]

    lr, n_steps = _lr_and_steps(m)
    log_every   = LOG_EVERY
    n_log       = n_steps // log_every + 1

    out_dir = os.path.join(FIG_BASE, target_key, f'm={m}', f'steps={n_steps}')
    f_traj  = os.path.join(out_dir, 'bias_trajectories.png')
    f_clust = os.path.join(out_dir, 'clusters_vs_inflections.png')
    f_csv   = os.path.join(out_dir, 'convergence_check.csv')
    f_meta  = os.path.join(out_dir, 'run_meta.csv')

    # ── Skip if already complete ──────────────────────────────────────────────
    if all(os.path.exists(p) for p in [f_traj, f_clust, f_csv, f_meta]):
        with open(f_meta, newline='') as mf:
            row = next(csv.DictReader(mf))
        return {
            'target':       target_key,
            'm':            m,
            'steps':        n_steps,
            'lr_eff':       lr,
            'k_true':       k_true,
            'n_clusters':   int(row['n_clusters']),
            'n_active':     int(row['n_active']),
            'loss':         float(row['loss']),
            'max_da_grad':  float(row['max_da_grad']),
            'max_db_grad':  float(row['max_db_grad']),
            'active_leq_k': int(row['active_leq_k']),
            '_skipped': True,
        }

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── Initialise ────────────────────────────────────────────────────────────
    rng    = np.random.default_rng(SEED)
    a      = rng.standard_normal(m) * 0.5
    b      = np.sort(rng.uniform(-1.0, 1.0, m))
    f_star = f_star_fn(X_QUAD)

    b_log    = np.zeros((n_log, m))
    loss_log = []
    b_log[0] = np.sort(b.copy())

    # ── Gradient descent loop ─────────────────────────────────────────────────
    for step in range(n_steps):
        H        = relu(X_QUAD[:, None] - b[None, :])          # (N_QUAD, m)
        f        = H @ a                                         # (N_QUAD,)
        residual = f - f_star                                    # (N_QUAD,)
        loss     = float(np.mean(residual ** 2))

        # dL/da_j = integral (f - f*) relu(x - b_j) dx
        da_grad = H.T @ residual * DX                           # (m,)

        # dL/db_j = -a_j * integral_{x > b_j} (f - f*) dx
        db_grad = -(a * ((X_QUAD[:, None] > b[None, :]).T @ residual * DX))

        a -= lr * da_grad
        b -= lr * db_grad

        if (step + 1) % log_every == 0:
            idx        = (step + 1) // log_every
            b_log[idx] = np.sort(b.copy())
            loss_log.append(loss)

    # ── Final metrics ─────────────────────────────────────────────────────────
    H_fin        = relu(X_QUAD[:, None] - b[None, :])
    f_fin        = H_fin @ a
    residual_fin = f_fin - f_star
    final_loss   = float(np.mean(residual_fin ** 2))

    da_grad_fin  = H_fin.T @ residual_fin * DX
    db_grad_fin  = -(a * ((X_QUAD[:, None] > b[None, :]).T @ residual_fin * DX))

    n_clusters      = count_clusters(b)
    cluster_centers = get_cluster_centers(b)
    a_scale         = max(float(np.max(np.abs(a))), 1e-12)
    n_active        = int(np.sum(np.abs(a) > ACTIVE_THRESHOLD * a_scale))
    active_leq_k    = int(n_active <= k_true)
    infl_x          = np.array(INFLECTIONS.get(target_key, []))

    lr_str  = f'{lr:.5f}'
    iters   = np.arange(n_log) * log_every

    # ── Figure 1: bias trajectories + final fit + loss ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'target={target_label},  m={m},  steps={n_steps},  lr={lr_str}'
        f'  |  clusters={n_clusters},  k={k_true}', fontsize=12)

    ax = axes[0]
    alpha = min(0.4, 20.0 / m)
    for j in range(m):
        ax.plot(iters, b_log[:, j], lw=0.4, alpha=alpha, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$b_j$ (kink location)')
    ax.set_title('Sorted Bias Trajectories')
    ax.set_xlim([0, n_steps])
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    x_plot = np.linspace(-1.0, 1.0, 500)
    f_plot = sum(a[j] * relu(x_plot - b[j]) for j in range(m))
    ax.plot(x_plot, f_star_fn(x_plot), 'k--', lw=2, label=f'Target {target_label}')
    ax.plot(x_plot, f_plot, 'r-', lw=2, label='Network $f$')
    y0, y1 = ax.get_ylim()
    tick_h = (y1 - y0) * 0.06
    ax.vlines(cluster_centers, y0, y0 + tick_h,
              colors='steelblue', linewidths=2.5, zorder=5)
    ax.set_xlabel('$x$')
    ax.set_title('Final Fit vs Target')
    ax.legend(fontsize=8)
    ax.set_xlim([-1, 1])

    ax = axes[2]
    if loss_log:
        ax.semilogy(np.arange(1, len(loss_log) + 1) * log_every, loss_log,
                    color='darkorange', lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Loss (log)   final={final_loss:.2e}')
    ax.set_xlim([0, n_steps])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f_traj, bbox_inches='tight', dpi=130)
    plt.close()

    # ── Figure 2: clusters vs inflection points ───────────────────────────────
    x_fine = np.linspace(-1.0, 1.0, 2000)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f'target={target_label},  m={m},  steps={n_steps},  lr={lr_str}  '
        f'|  clusters={n_clusters},  k={k_true}  '
        f'(C=k: {n_clusters == k_true})', fontsize=11)
    ax.plot(x_fine, f_star_fn(x_fine), 'k-', lw=2, label=f'Target {target_label}')
    ax.plot(x_plot, f_plot, 'r-', lw=1.5, label='Final network')
    for cx in cluster_centers:
        ax.axvline(cx, color='steelblue', alpha=0.5, lw=0.7)
    for ix in infl_x:
        ax.axvline(ix, color='green', alpha=0.7, lw=1.2, linestyle='--')
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color='steelblue', lw=1.5,
               label=f'Bias clusters ({n_clusters})'),
        Line2D([0], [0], color='green', lw=1.5, ls='--',
               label=f'Inflection pts k={k_true}'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('$x$')
    ax.set_title('Cluster Locations vs Inflection Points')
    plt.tight_layout()
    plt.savefig(f_clust, bbox_inches='tight', dpi=130)
    plt.close()

    # ── convergence_check.csv ─────────────────────────────────────────────────
    with open(f_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['b_j', 'a_j', 'da_grad', 'db_grad', 'active'])
        for j in range(m):
            active_j = int(abs(a[j]) > ACTIVE_THRESHOLD * a_scale)
            writer.writerow([b[j], a[j], da_grad_fin[j], db_grad_fin[j], active_j])

    # ── run_meta.csv ──────────────────────────────────────────────────────────
    result = {
        'target':       target_key,
        'm':            m,
        'steps':        n_steps,
        'lr_eff':       lr,
        'k_true':       k_true,
        'n_clusters':   n_clusters,
        'n_active':     n_active,
        'loss':         final_loss,
        'max_da_grad':  float(np.max(np.abs(da_grad_fin))),
        'max_db_grad':  float(np.max(np.abs(db_grad_fin))),
        'active_leq_k': active_leq_k,
        '_elapsed':     time.time() - t0,
    }

    with open(f_meta, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerow({k: result[k] for k in META_FIELDS})

    return result


# =============================================================================
# Convergence plot
# =============================================================================

def make_convergence_plot(rows, out_path=CONV_PLOT):
    targets_present = list(dict.fromkeys(
        k for k in TARGETS if any(r['target'] == k for r in rows)
    ))
    ncols = 3
    nrows = -(-len(targets_present) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, tkey in enumerate(targets_present):
        ax     = axes_flat[ax_i]
        k_true = TARGETS[tkey][1]
        pts    = sorted([r for r in rows if r['target'] == tkey],
                        key=lambda r: r['m'])
        if pts:
            ms = [r['m']          for r in pts]
            cs = [r['n_clusters'] for r in pts]
            ax.plot(ms, cs, marker='o', color='steelblue',
                    lw=1.5, ms=5, label='Discrete GD (scaled lr)')
        ax.axhline(k_true, color='crimson', lw=2, linestyle='--',
                   label=f'k = {k_true}')
        ax.set_xlabel('Width $m$', fontsize=10)
        ax.set_ylabel('Cluster count $C(m, f^*)$', fontsize=10)
        ax.set_title(TARGETS[tkey][0], fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax_i in range(len(targets_present), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    lr_note = (f'lr = {LR_BASE}/√(m/50)  [= {LR_BASE} at m=50]'
               if LR_SCALE else f'lr = {LR_BASE} (constant)')
    fig.suptitle(
        f'Open Problem 4.1 — Discrete GD  ({lr_note})\n'
        f'Crimson dashed = analytical $k$   ({len(rows)} completed runs)',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Convergence plot -> {out_path}')


# =============================================================================
# Helper: load existing run_summary_discrete.csv
# =============================================================================

def load_summary(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'target':       row['target'],
                'm':            int(row['m']),
                'steps':        int(row['steps']),
                'lr_eff':       float(row.get('lr_eff', LR_BASE)),
                'k_true':       int(row['k_true']),
                'n_clusters':   int(row['n_clusters']),
                'n_active':     int(row['n_active']),
                'loss':         float(row['loss']),
                'max_da_grad':  float(row['max_da_grad']),
                'max_db_grad':  float(row['max_db_grad']),
                'active_leq_k': int(row['active_leq_k']),
            })
    return rows


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(FIG_BASE, exist_ok=True)

    # Pre-compute Phase 2 m sequence for header print
    _phase2_m_seq = list(PHASE2_M_FIXED)
    _m = max(PHASE2_M_FIXED)
    while _m < PHASE2_M_MAX:
        _m = min(_m + PHASE2_M_INCREMENT, PHASE2_M_MAX)
        if _m not in _phase2_m_seq:
            _phase2_m_seq.append(_m)

    existing_rows = load_summary(SUMMARY_CSV)
    n_workers     = max(1, min(cpu_count() - 2, 20))

    base_jobs = sorted(
        [(t, m) for t in TARGETS for m in M_VALUES_ALL],
        key=lambda x: (x[1], x[0])   # m first, then target name
    )

    lr_note = (f'lr = {LR_BASE}/√(m/50)  [= {LR_BASE:.4g} at m=50, '
               f'{LR_BASE/np.sqrt(1000/50):.4g} at m=1000]'
               if LR_SCALE else f'lr = {LR_BASE} (constant)')

    print('=' * 72)
    print('simulate_discrete.py — Open Problem 4.1  (Discrete Gradient Descent)')
    print(f'Mode         : {"SCALED lr (1/sqrt(m/50))" if LR_SCALE else "CONSTANT lr"}')
    print(f'Learning rate: {lr_note}')
    print(f'Steps        : {N_STEPS_BASE} at m=50'
          + (f', up to {N_STEPS_MAX} (scales with sqrt(m/50))'
             if LR_SCALE else ' (constant)'))
    print(f'Targets      : {list(TARGETS.keys())}')
    print(f'Base m       : {M_VALUES_ALL}')
    print(f'Extended m   : {_phase2_m_seq}  (Phase 2, non-converged only)')
    print(f'Phase 1 jobs : {len(base_jobs)}')
    print(f'Workers      : {n_workers}  (of {cpu_count()} logical CPUs)')
    print(f'Previous rows: {len(existing_rows)}')
    print(f'Output dir   : {FIG_BASE}')
    print(f'Summary  ->  {SUMMARY_CSV}')
    print(f'Plot     ->  {CONV_PLOT}')
    print('=' * 72)

    t_start   = time.time()
    all_new   = []
    skipped   = 0
    completed = 0

    # ── Helper: submit a list of (target, m) jobs and collect results ─────────
    def run_phase(jobs_list, phase_label):
        phase_rows      = []
        phase_skipped   = 0
        phase_completed = 0
        print(f'\n{"=" * 72}')
        print(phase_label)
        print(f'{"=" * 72}')
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(run_one, jobs_list):
                if result is None:
                    phase_skipped += 1
                    continue
                was_skipped = result.pop('_skipped', False)
                elapsed     = result.pop('_elapsed', None)
                phase_rows.append(result)
                tag = (f'{result["target"]:<12}  m={result["m"]:<5}  '
                       f'clusters={result["n_clusters"]:<4}  k={result["k_true"]}  '
                       f'C=k: {result["n_clusters"] == result["k_true"]}  '
                       f'loss={result["loss"]:.3e}  '
                       f'lr={result["lr_eff"]:.5f}  '
                       f'steps={result["steps"]}  '
                       f'max|db|={result["max_db_grad"]:.2e}')
                if was_skipped:
                    phase_skipped += 1
                    print(f'  SKIP (meta)  {tag}')
                else:
                    phase_completed += 1
                    t_str = f'  [{elapsed:.0f}s]' if elapsed else ''
                    print(f'  DONE{t_str}  {tag}')
        return phase_rows, phase_skipped, phase_completed

    # ── Phase 1: base m sweep ─────────────────────────────────────────────────
    p1_rows, p1_skip, p1_done = run_phase(
        base_jobs, 'Phase 1 — base m sweep (all targets)')
    all_new.extend(p1_rows)
    skipped   += p1_skip
    completed += p1_done

    # ── Phase 1 → Phase 2 convergence check at max base m ────────────────────
    MAX_BASE_M  = max(M_VALUES_ALL)
    all_results = {(r['target'], r['m']): r for r in existing_rows}
    for r in all_new:
        all_results[(r['target'], r['m'])] = r

    non_converged = []
    print(f'\n── Convergence check at m={MAX_BASE_M}  '
          f'(threshold |C-k| <= {CONV_THRESHOLD}) {"─" * 20}')
    for t_key, (_, k_true, _) in TARGETS.items():
        at_max = [v for (t, m), v in all_results.items()
                  if t == t_key and m == MAX_BASE_M]
        if not at_max:
            print(f'  {t_key:<12}  k={k_true}  NO DATA — queuing Phase 2')
            non_converged.append(t_key)
        elif any(abs(r['n_clusters'] - k_true) <= CONV_THRESHOLD for r in at_max):
            best = min(abs(r['n_clusters'] - k_true) for r in at_max)
            print(f'  {t_key:<12}  k={k_true}  CONVERGED  (min |C-k| = {best})')
        else:
            diffs = [abs(r['n_clusters'] - k_true) for r in at_max]
            print(f'  {t_key:<12}  k={k_true}  NOT CONVERGED  '
                  f'(min |C-k| = {min(diffs)}) — queuing Phase 2')
            non_converged.append(t_key)

    # ── Phase 2: extended m, per-target 2-consecutive-hit early stop ──────────
    if non_converged:
        phase2_m_seq = list(PHASE2_M_FIXED)
        _m = max(PHASE2_M_FIXED)
        while _m < PHASE2_M_MAX:
            _m = min(_m + PHASE2_M_INCREMENT, PHASE2_M_MAX)
            if _m not in phase2_m_seq:
                phase2_m_seq.append(_m)

        print(f'\nPhase 2 targets    : {non_converged}')
        print(f'Phase 2 m sequence : {phase2_m_seq}')
        print(f'Convergence rule   : |C-k| <= {CONV_THRESHOLD} for 2 consecutive m values')
        print(f'Hard stop          : m = {PHASE2_M_MAX}')

        remaining   = list(non_converged)
        consec_hits = {t: 0 for t in non_converged}

        for m_val in phase2_m_seq:
            if not remaining:
                break

            batch_jobs = sorted(
                [(t, m_val) for t in remaining],
                key=lambda x: x[0]
            )
            p2_rows, p2_skip, p2_done = run_phase(
                batch_jobs,
                f'Phase 2 — m={m_val}  ({len(remaining)} targets remaining)'
            )
            all_new.extend(p2_rows)
            skipped   += p2_skip
            completed += p2_done

            for r in p2_rows:
                all_results[(r['target'], r['m'])] = r

            still_remaining = []
            print(f'\n── Phase 2 convergence check at m={m_val} {"─" * 30}')
            for t_key in remaining:
                _, k_true, _ = TARGETS[t_key]
                at_m = [v for (t, m), v in all_results.items()
                        if t == t_key and m == m_val]
                hit_this_m = at_m and any(
                    abs(r['n_clusters'] - k_true) <= CONV_THRESHOLD for r in at_m
                )

                if hit_this_m:
                    consec_hits[t_key] += 1
                else:
                    consec_hits[t_key] = 0

                hits = consec_hits[t_key]

                if hits >= 2:
                    print(f'  {t_key:<12}  k={k_true}  CONVERGED  '
                          f'(|C-k| <= {CONV_THRESHOLD} for 2 consecutive m) — done')
                elif hits == 1:
                    best = min(abs(r['n_clusters'] - k_true) for r in at_m)
                    print(f'  {t_key:<12}  k={k_true}  1st hit at m={m_val}  '
                          f'(|C-k| = {best}) — need 1 more consecutive')
                    still_remaining.append(t_key)
                else:
                    diffs    = [abs(r['n_clusters'] - k_true) for r in at_m] if at_m else []
                    diff_str = str(min(diffs)) if diffs else 'no data'
                    if m_val == PHASE2_M_MAX:
                        print(f'  {t_key:<12}  k={k_true}  HARD STOP at m={PHASE2_M_MAX}  '
                              f'(min |C-k| = {diff_str}) — not converged')
                    else:
                        print(f'  {t_key:<12}  k={k_true}  not converged  '
                              f'(min |C-k| = {diff_str}) — continuing')
                        still_remaining.append(t_key)

            remaining = still_remaining

        not_conv = [t for t in non_converged if consec_hits[t] < 2]
        if not_conv:
            print(f'\nPhase 2 complete (hard stop m={PHASE2_M_MAX}). '
                  f'Did not converge: {not_conv}')
        else:
            print('\nAll Phase 2 targets confirmed converged (2 consecutive hits).')
    else:
        print('\nAll targets converged at base m — Phase 2 not needed.')

    # ── Write run_summary_discrete.csv ───────────────────────────────────────
    merged = {(r['target'], r['m']): r for r in existing_rows}
    for r in all_new:
        merged[(r['target'], r['m'])] = r
    final_rows = sorted(merged.values(), key=lambda r: (r['target'], r['m']))

    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for r in final_rows:
            writer.writerow({k: r.get(k, '') for k in META_FIELDS})
    print(f'\nSummary -> {SUMMARY_CSV}  ({len(final_rows)} rows)')

    # ── Convergence plot ──────────────────────────────────────────────────────
    if final_rows:
        make_convergence_plot(final_rows)

    wall = time.time() - t_start
    print(f'\nDone.  {completed} new runs,  {skipped} skipped,  '
          f'wall time {wall / 60:.1f} min')
