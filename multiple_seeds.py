"""
multiple_seeds.py
=================
Re-run the four below-k stationary cases with additional random seeds
to determine whether below-k states are:
  (A) Genuine fixed points of the ODE for this target/m combination
      regardless of initialization, OR
  (B) Initialization-dependent local minima — only one basin of attraction.

The three cases (all from simulate_parallel M_VALUES, C < k at seed=42, T=500):
    sin_2pi  m=1000  k=3  C=2
    sin_4pi  m=2000  k=7  C=2
    poly_k3  m=5000  k=3  C=2

sin_3pi m=1500 excluded: m=1500 is not in simulate_parallel M_VALUES.

Seeds tested here: 5, 25  (seed=42 is the existing simulate_parallel result)
Total new runs: 3 cases × 2 seeds = 6 runs

Output
------
  Per run:
      figures/Replication data/{target}/m={m}/T=500/seed={seed}/
          run_meta.csv          (same schema as simulate_parallel)
          convergence_check.csv (one row per neuron with a_j, b_j)
  Summary:
      figures/Replication data/multiple_seeds_results.csv
      figures/Replication data/multiple_seeds_summary.png

Usage
-----
    # Run from project root (MathProject4/)
    python multiple_seeds.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Parameters — identical to simulate_parallel.py
# =============================================================================
T_FINAL          = 500
N_QUAD           = 200
X_QUAD           = np.linspace(-1.0, 1.0, N_QUAD)
DX               = X_QUAD[1] - X_QUAD[0]
N_SAVE           = 300
ACTIVE_THRESHOLD = 0.05
CLUSTER_TOL      = 0.02
FIG_BASE         = os.path.join('figures', 'Replication data')

SUMMARY_CSV = os.path.join(FIG_BASE, 'multiple_seeds_results.csv')
SUMMARY_PNG = os.path.join(FIG_BASE, 'multiple_seeds_summary.png')

# Existing seed=42 results (from simulate_parallel.py)
SEED_REFERENCE = 42

# Additional seeds to test
SEEDS = [5, 25]

# The three below-k cases to investigate (all from simulate_parallel M_VALUES).
# sin_3pi m=1500 dropped: m=1500 was not in M_VALUES so that run came from a
# legacy sweep with unclear provenance.  No simulate_parallel run of sin_3pi
# shows genuine below-k (m=1000 → C=k; m≥2000 → dense-packing artifact).
CASES = [
    ('sin_2pi', 1000,  3),   # C=2 at seed=42
    ('sin_4pi', 2000,  7),   # C=2 at seed=42
    ('poly_k3', 5000,  3),   # C=2 at seed=42
]

META_FIELDS = [
    'target', 'm', 'T', 'seed', 'k_true',
    'n_clusters', 'n_active', 'loss',
    'max_da', 'max_db', 'active_leq_k',
]

# =============================================================================
# Core math (matches simulate_parallel.py exactly)
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def network(x, a, b):
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def make_ode(m, f_star_fn):
    fstar_vals = f_star_fn(X_QUAD)
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

def compute_ode_velocities(a, b, f_star_fn):
    fstar_vals = f_star_fn(X_QUAD)
    residual   = network(X_QUAD, a, b) - fstar_vals
    relu_mat   = relu(X_QUAD[:, None] - b[None, :])
    da         = -(residual[:, None] * relu_mat).sum(0) * DX
    cum_right  = np.cumsum(residual[::-1])[::-1] * DX
    idx        = np.searchsorted(X_QUAD, b).clip(0, N_QUAD - 1)
    db         = a * cum_right[idx]
    return da, db

# =============================================================================
# Target functions (named defs for multiprocessing pickle safety)
# =============================================================================

def f_sin_2pi(x): return np.sin(2 * np.pi * x)
def f_sin_3pi(x): return np.sin(3 * np.pi * x)
def f_sin_4pi(x): return np.sin(4 * np.pi * x)
def f_poly_k3(x): return x**5 - 3 * x**3

TARGET_FNS = {
    'sin_2pi': (r'$\sin(2\pi x)$', f_sin_2pi),
    'sin_3pi': (r'$\sin(3\pi x)$', f_sin_3pi),
    'sin_4pi': (r'$\sin(4\pi x)$', f_sin_4pi),
    'poly_k3': (r'$x^5-3x^3$',     f_poly_k3),
}

# =============================================================================
# Single-run worker
# =============================================================================

def run_one(args):
    """
    Run one (target_key, m, k_true, seed) combination.
    Returns a metrics dict, or None if skipped (already complete).
    """
    target_key, m, k_true, seed = args
    target_label, f_star = TARGET_FNS[target_key]

    out_dir  = os.path.join(FIG_BASE, target_key, f'm={m}', 'T=500', f'seed={seed}')
    f_meta   = os.path.join(out_dir, 'run_meta.csv')
    f_check  = os.path.join(out_dir, 'convergence_check.csv')

    # ── Skip if already complete ──────────────────────────────────────────────
    if os.path.exists(f_meta) and os.path.exists(f_check):
        with open(f_meta, newline='') as mf:
            row = next(csv.DictReader(mf))
        print(f'  SKIP  {target_key:<12} m={m:<5} seed={seed}  '
              f'C={row["n_clusters"]}  k={k_true}')
        return {
            'target':       target_key,
            'm':            m,
            'T':            T_FINAL,
            'seed':         seed,
            'k_true':       k_true,
            'n_clusters':   int(row['n_clusters']),
            'n_active':     int(row['n_active']),
            'loss':         float(row['loss']),
            'max_da':       float(row['max_da']),
            'max_db':       float(row['max_db']),
            'active_leq_k': int(row['active_leq_k']),
            '_skipped': True,
        }

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── Initialization (same as simulate_parallel, different seed) ────────────
    np.random.seed(seed)
    a0 = np.random.randn(m) * 0.01
    b0 = np.random.uniform(-1.0, 1.0, m)

    # ── ODE integration ───────────────────────────────────────────────────────
    sol = solve_ivp(
        make_ode(m, f_star),
        t_span=(0.0, T_FINAL),
        y0=np.concatenate([a0, b0]),
        method='RK45',
        t_eval=np.linspace(0.0, T_FINAL, N_SAVE),
        rtol=1e-4, atol=1e-6,
        max_step=max(0.1, T_FINAL / 500),
    )
    elapsed = time.time() - t0

    # ── Final state ───────────────────────────────────────────────────────────
    a_final    = sol.y[:m, -1]
    b_final    = sol.y[m:, -1]
    n_clusters = count_clusters(b_final)

    fstar_vals = f_star(X_QUAD)
    final_loss = float(0.5 * np.trapz(
        (network(X_QUAD, a_final, b_final) - fstar_vals) ** 2, X_QUAD))

    da_final, db_final = compute_ode_velocities(a_final, b_final, f_star)
    max_da = float(np.max(np.abs(da_final)))
    max_db = float(np.max(np.abs(db_final)))

    a_max      = max(float(np.abs(a_final).max()), 1e-12)
    is_active  = np.abs(a_final) > ACTIVE_THRESHOLD * a_max
    n_active   = int(is_active.sum())
    active_leq = int(n_active <= k_true)

    # ── Save run_meta.csv ─────────────────────────────────────────────────────
    with open(f_meta, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerow({
            'target':       target_key,
            'm':            m,
            'T':            T_FINAL,
            'seed':         seed,
            'k_true':       k_true,
            'n_clusters':   n_clusters,
            'n_active':     n_active,
            'loss':         f'{final_loss:.10f}',
            'max_da':       f'{max_da:.6f}',
            'max_db':       f'{max_db:.6f}',
            'active_leq_k': active_leq,
        })

    # ── Save convergence_check.csv (same format as simulate_parallel) ─────────
    with open(f_check, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['neuron_idx', 'a_j', 'b_j'])
        writer.writeheader()
        for j in range(m):
            writer.writerow({
                'neuron_idx': j,
                'a_j':        f'{a_final[j]:.8f}',
                'b_j':        f'{b_final[j]:.8f}',
            })

    print(f'  DONE  {target_key:<12} m={m:<5} seed={seed}  '
          f'C={n_clusters}  k={k_true}  '
          f'max_da={max_da:.4f}  '
          f'{"stationary" if max_da < 0.01 else "NOT stationary"}  '
          f'({elapsed:.1f}s)')

    return {
        'target':       target_key,
        'm':            m,
        'T':            T_FINAL,
        'seed':         seed,
        'k_true':       k_true,
        'n_clusters':   n_clusters,
        'n_active':     n_active,
        'loss':         final_loss,
        'max_da':       max_da,
        'max_db':       max_db,
        'active_leq_k': active_leq,
    }

# =============================================================================
# Summary figure
# =============================================================================

def make_summary(results, reference_c):
    """
    Bar chart: for each case, show C at seed=42, seed=1, seed=2 side by side.
    reference_c: dict mapping (target, m) -> C value from seed=42 run.
    """
    cases_order = [(t, m) for t, m, _ in CASES]
    case_labels  = [f'{t}\nm={m}' for t, m in cases_order]
    k_vals       = {(t, m): k for t, m, k in CASES}

    seed_all  = [SEED_REFERENCE] + SEEDS
    bar_width = 0.22
    x         = np.arange(len(cases_order))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, seed in enumerate(seed_all):
        c_vals = []
        for (t, m) in cases_order:
            if seed == SEED_REFERENCE:
                c_vals.append(reference_c.get((t, m), float('nan')))
            else:
                match = [r for r in results if r['target'] == t
                         and r['m'] == m and r['seed'] == seed]
                c_vals.append(match[0]['n_clusters'] if match else float('nan'))
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, c_vals, width=bar_width,
                      color=colors[i], edgecolor='black',
                      label=f'seed={seed}', alpha=0.85)
        for bar, cv in zip(bars, c_vals):
            if not np.isnan(cv):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(int(cv)), ha='center', va='bottom', fontsize=9)

    # Draw k lines per case
    for i, (t, m) in enumerate(cases_order):
        k = k_vals[(t, m)]
        ax.hlines(k, x[i] - 1.5 * bar_width, x[i] + 1.5 * bar_width,
                  colors='crimson', linestyles='--', lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=10)
    ax.set_ylabel('Cluster count $C$')
    ax.set_title('Below-k Cases: Cluster Count Across Seeds\n'
                 '(crimson dashed = k, target cluster count)')
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(10, ax.get_ylim()[1] + 1)])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Summary figure -> {SUMMARY_PNG}')

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()

    print('multiple_seeds.py')
    print(f'Testing seeds: {SEEDS}  (reference seed={SEED_REFERENCE} from simulate_parallel)')
    print(f'Cases ({len(CASES)}):')
    for t, m, k in CASES:
        print(f'  {t:<12}  m={m:<5}  k={k}  (C=2 at seed={SEED_REFERENCE})')
    print()

    # ── Read existing seed=42 cluster counts from run_meta.csv ───────────────
    reference_c = {}
    for target_key, m, k_true in CASES:
        meta_path = os.path.join(FIG_BASE, target_key, f'm={m}', 'T=500', 'run_meta.csv')
        if os.path.exists(meta_path):
            with open(meta_path, newline='') as f:
                row = next(csv.DictReader(f))
            reference_c[(target_key, m)] = int(row['n_clusters'])
            print(f'  seed=42 reference  {target_key:<12} m={m}  C={reference_c[(target_key, m)]}')
        else:
            print(f'  WARNING: no seed=42 run_meta.csv found for {target_key} m={m}')
    print()

    # ── Build job list ────────────────────────────────────────────────────────
    jobs = []
    for target_key, m, k_true in CASES:
        for seed in SEEDS:
            jobs.append((target_key, m, k_true, seed))

    print(f'Jobs to run: {len(jobs)}')

    n_workers = max(1, min(cpu_count() - 2, len(jobs)))
    print(f'Workers    : {n_workers}')
    print()

    # ── Run in parallel ───────────────────────────────────────────────────────
    results = []
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_one, jobs):
            if result is not None and not result.get('_skipped', False):
                results.append(result)
            elif result is not None and result.get('_skipped', False):
                results.append(result)   # include skipped for summary

    # ── Write summary CSV ─────────────────────────────────────────────────────
    sorted_results = sorted(results,
                            key=lambda r: (r['target'], r['m'], r['seed']))
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow({k: r[k] for k in META_FIELDS})
    print(f'\nResults CSV -> {SUMMARY_CSV}')

    # ── Print comparison table ────────────────────────────────────────────────
    print()
    print(f'{"Target":<12} {"m":>5}  {"k":>2}  '
          f'{"seed=42":>8}  {"seed=1":>8}  {"seed=2":>8}  Verdict')
    print('-' * 72)
    for target_key, m, k_true in CASES:
        c42   = reference_c.get((target_key, m), '?')
        c_by_seed = {}
        for r in sorted_results:
            if r['target'] == target_key and r['m'] == m:
                c_by_seed[r['seed']] = r['n_clusters']
        c1 = c_by_seed.get(1, '?')
        c2 = c_by_seed.get(2, '?')
        all_c = [v for v in [c42, c1, c2] if isinstance(v, int)]
        if all_c and all(c == all_c[0] for c in all_c):
            verdict = 'FIXED POINT (all seeds agree)'
        elif k_true in all_c:
            verdict = f'LOCAL MINIMUM (some seeds reach k={k_true})'
        else:
            verdict = 'INCONCLUSIVE'
        print(f'{target_key:<12} {m:>5}  {k_true:>2}  '
              f'{str(c42):>8}  {str(c1):>8}  {str(c2):>8}  {verdict}')

    # ── Summary figure ────────────────────────────────────────────────────────
    make_summary(sorted_results, reference_c)

    wall = time.time() - t_start
    print(f'\nDone.  Wall time: {wall:.1f}s')
