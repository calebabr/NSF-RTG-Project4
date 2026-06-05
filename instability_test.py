"""
instability_test.py
===================
Goal 2: Show numerically that k+1 cluster configurations are unstable.

Background
----------
The slides establish that the k-cluster merged state is neutrally stable:
the first-order gradient along the separation direction is zero once biases
merge. Goal 2 asks the converse: if you START with k+1 clusters (one more
than the optimal k), does gradient flow drive the system back to k clusters?
If yes, that instability is often easier to prove mathematically than
convergence itself, and gives a concrete mechanism for a future proof.

What this script does
---------------------
For every run folder where the simulation converged to exactly k clusters
(n_clusters == k_true from run_meta.csv), this script:

  1. Loads the final converged state (b_j, a_j) from convergence_check.csv
  2. Injects one extra neuron, creating a k+1 cluster configuration.
     Two injection strategies are tested per run:
       'near'     -- inject just outside an existing cluster boundary.
                     Tests whether proximity causes the neuron to merge in.
       'isolated' -- inject at the location farthest from all cluster centers.
                     Tests whether the amplitude decays when there is no
                     nearby cluster to merge with.
  3. Continues integrating the ODE forward from the perturbed state for
     T_PERTURB time units using the same gradient flow equations.
  4. Tracks cluster count C(t), the injected neuron amplitude |a_inj(t)|,
     and the injected neuron bias location b_inj(t) over time.
  5. Records whether the system returned to k clusters by T_PERTURB.

Run order
---------
  1. python simulate.py
  2. python simulate_parallel.py
  3. python verify_pruning.py
  4. python instability_test.py   <-- this script

Outputs
-------
  Per qualifying run and injection type
  (saved in the original run folder):
      goal2_{injection_type}.png   -- 3-panel figure tracking the perturbation

  Global (saved in figures/Replication data/):
      goal2_results.csv            -- one row per (run, injection type)
      goal2_summary.png            -- summary across all tested runs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Constants  (must match main scripts)
# =============================================================================
N_QUAD      = 400
X_QUAD      = np.linspace(-1.0, 1.0, N_QUAD)
DX          = X_QUAD[1] - X_QUAD[0]
CLUSTER_TOL = 0.02
T_PERTURB   = 1000      # integration time after injection
N_SAVE      = 300       # trajectory snapshots during perturbation
A_INJECT_SCALE = 0.1   # injected amplitude = A_INJECT_SCALE * mean|a_j|

FIG_BASE    = os.path.join('figures', 'Replication data')
RESULTS_CSV = os.path.join(FIG_BASE, 'goal2_results.csv')
SUMMARY_PNG = os.path.join(FIG_BASE, 'goal2_summary.png')

RESULTS_FIELDS = [
    'target', 'm', 'T_original', 'k_true',
    'injection_type',
    'b_inject',          # where the neuron was injected
    'a_inject_initial',  # amplitude at injection
    'a_inject_final',    # amplitude at end of perturbation
    'a_decayed',         # 1 if |a_inject_final| < 0.1 * |a_inject_initial|
    'b_inject_final',    # bias location at end of perturbation
    'b_merged',          # 1 if final bias is within CLUSTER_TOL of any original cluster
    'final_n_clusters',  # cluster count at end of perturbation
    'returned_to_k',     # 1 if final_n_clusters == k_true
]

# =============================================================================
# Target functions  (re-defined for self-containment; lambdas looked up inside
# worker after module re-import, so no pickling of functions needed)
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

def count_clusters(biases, tol=CLUSTER_TOL):
    s = np.sort(biases)
    return 1 + int(np.sum(np.diff(s) > tol))

def get_cluster_centers(b, a, tol=CLUSTER_TOL):
    sort_idx = np.argsort(b)
    bs, as_ = b[sort_idx], a[sort_idx]
    centers, group_b = [], [bs[0]]
    for i in range(1, len(bs)):
        if bs[i] - bs[i-1] > tol:
            centers.append(float(np.mean(group_b)))
            group_b = []
        group_b.append(bs[i])
    centers.append(float(np.mean(group_b)))
    return np.array(centers)

# =============================================================================
# Injection location strategies
# =============================================================================

def inject_near(cluster_centers):
    """
    Place the new neuron just outside the boundary of the cluster that has
    the most room on one side. Specifically: find the largest gap between
    adjacent cluster centers (or between the domain boundary and the nearest
    cluster), then place the neuron at 3 * CLUSTER_TOL from the nearest
    cluster edge — close enough to interact but not already merged.
    """
    # Include domain boundaries as virtual cluster centers for gap analysis
    pts = np.concatenate([[-1.0], np.sort(cluster_centers), [1.0]])
    gaps = np.diff(pts)
    idx  = int(np.argmax(gaps))     # largest gap index
    left, right = pts[idx], pts[idx + 1]
    # Place near the cluster center that is closest to the gap midpoint
    midpoint = (left + right) / 2.0
    # Pick the cluster center nearest to the midpoint
    nearest  = cluster_centers[np.argmin(np.abs(cluster_centers - midpoint))]
    # Inject at 3 * tol away from nearest cluster, toward midpoint
    direction = np.sign(midpoint - nearest) if midpoint != nearest else 1.0
    b_inject  = nearest + direction * 3 * CLUSTER_TOL
    b_inject  = float(np.clip(b_inject, -1.0 + 1e-4, 1.0 - 1e-4))
    return b_inject

def inject_isolated(cluster_centers):
    """
    Place the new neuron at the location in [-1, 1] that is farthest from
    all existing cluster centers. This is the maximally isolated location,
    where the injected neuron has the least interaction with existing clusters.
    """
    candidates = np.linspace(-1.0, 1.0, 2000)
    min_dists  = np.array([np.min(np.abs(x - cluster_centers))
                           for x in candidates])
    b_inject = float(candidates[np.argmax(min_dists)])
    return b_inject

# =============================================================================
# Load converged state from run folder
# =============================================================================

def load_converged_run(run_dir):
    """
    Load b_j, a_j from convergence_check.csv and metadata from run_meta.csv.
    Returns None if files are missing or the run did not converge to k clusters.
    """
    f_csv  = os.path.join(run_dir, 'convergence_check.csv')
    f_meta = os.path.join(run_dir, 'run_meta.csv')
    if not (os.path.exists(f_csv) and os.path.exists(f_meta)):
        return None

    with open(f_meta, newline='') as mf:
        meta = next(csv.DictReader(mf))

    n_clusters = int(meta['n_clusters'])
    k_true     = int(meta['k_true'])
    if n_clusters != k_true:
        return None   # not converged to k; skip

    b_vals, a_vals = [], []
    with open(f_csv, newline='') as cf:
        for row in csv.DictReader(cf):
            b_vals.append(float(row['b_j']))
            a_vals.append(float(row['a_j']))

    return {
        'target':     meta['target'],
        'm':          int(meta['m']),
        'T_original': int(meta['T']),
        'k_true':     k_true,
        'b':          np.array(b_vals),
        'a':          np.array(a_vals),
    }

# =============================================================================
# Worker: test one (run, injection_type) combination
# =============================================================================

def test_one(args):
    run_dir, run_data, injection_type = args

    target_key  = run_data['target']
    m_orig      = run_data['m']
    T_original  = run_data['T_original']
    k_true      = run_data['k_true']
    b_orig      = run_data['b']
    a_orig      = run_data['a']

    if target_key not in TARGETS:
        return None
    target_label, _, f_star = TARGETS[target_key]

    out_fig = os.path.join(run_dir, f'goal2_{injection_type}.png')

    # ── Compute injection location ────────────────────────────────────────────
    cluster_centers = get_cluster_centers(b_orig, a_orig)
    if injection_type == 'near':
        b_inject = inject_near(cluster_centers)
    else:
        b_inject = inject_isolated(cluster_centers)

    # Injected amplitude: small relative to the network scale
    a_inject_initial = float(A_INJECT_SCALE * np.abs(a_orig).mean())
    a_inject_initial = max(a_inject_initial, 0.01)   # at least 0.01

    # ── Build perturbed initial state  (injected neuron appended last) ────────
    m_new  = m_orig + 1
    a_init = np.append(a_orig, a_inject_initial)
    b_init = np.append(b_orig, b_inject)
    y0     = np.concatenate([a_init, b_init])

    # ── Integrate forward ─────────────────────────────────────────────────────
    sol = solve_ivp(
        make_ode(m_new, f_star),
        t_span=(0.0, T_PERTURB),
        y0=y0,
        method='RK45',
        t_eval=np.linspace(0.0, T_PERTURB, N_SAVE),
        rtol=1e-4, atol=1e-6,
        max_step=max(0.1, T_PERTURB / 500),
    )

    # ── Extract trajectories for the injected neuron (index m_orig) ───────────
    a_inj_traj = sol.y[m_orig, :]            # amplitude of injected neuron
    b_inj_traj = sol.y[m_new + m_orig, :]   # bias of injected neuron

    # ── Cluster count over time ───────────────────────────────────────────────
    cluster_counts = np.array([
        count_clusters(sol.y[m_new:, i]) for i in range(sol.y.shape[1])
    ])

    # ── Final state metrics ───────────────────────────────────────────────────
    a_inject_final = float(a_inj_traj[-1])
    b_inject_final = float(b_inj_traj[-1])
    final_n_clusters = int(cluster_counts[-1])

    # Did the amplitude decay? (dropped to less than 10% of initial)
    a_decayed = int(abs(a_inject_final) < 0.1 * abs(a_inject_initial))

    # Did the bias merge into an existing cluster?
    b_merged = int(np.min(np.abs(b_inject_final - cluster_centers)) < CLUSTER_TOL)

    returned_to_k = int(final_n_clusters == k_true)

    # ── Figure ────────────────────────────────────────────────────────────────
    if not os.path.exists(out_fig):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Goal 2: Instability of k+1 Clusters  ({injection_type} injection)\n'
            f'target={target_label},  m={m_orig},  T_orig={T_original},  k={k_true}  '
            f'|  returned to k: {bool(returned_to_k)}',
            fontsize=10)

        # Panel 1: sorted bias trajectories (injected neuron in red)
        ax = axes[0]
        # Original neurons — sort each snapshot
        for j in range(m_orig):
            traj = np.sort(sol.y[m_new:m_new + m_orig, :], axis=0)[j]
            ax.plot(sol.t, traj, color='steelblue',
                    alpha=min(0.3, 15.0 / m_orig), linewidth=0.5)
        ax.plot(sol.t, b_inj_traj, color='crimson', lw=1.8,
                label='Injected neuron')
        for cc in cluster_centers:
            ax.axhline(cc, color='green', lw=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('Bias location')
        ax.set_title('Bias Trajectories\n(red = injected, green = original clusters)')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])

        # Panel 2: injected neuron amplitude over time
        ax = axes[1]
        ax.plot(sol.t, np.abs(a_inj_traj), color='darkorange', lw=1.8)
        ax.axhline(0.1 * abs(a_inject_initial), color='k', lw=1,
                   linestyle='--', label='10% of initial (decay threshold)')
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('$|a_{\\mathrm{inject}}(t)|$')
        ax.set_title(f'Injected Amplitude\ninitial={a_inject_initial:.3f},  '
                     f'final={abs(a_inject_final):.3f},  decayed: {bool(a_decayed)}')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])
        ax.grid(True, alpha=0.3)

        # Panel 3: cluster count over time
        ax = axes[2]
        ax.plot(sol.t, cluster_counts, color='purple', lw=1.8)
        ax.axhline(k_true,     color='crimson', lw=1.5, linestyle='--',
                   label=f'k={k_true} (target)')
        ax.axhline(k_true + 1, color='gray',   lw=1.0, linestyle=':',
                   label=f'k+1={k_true+1} (injected)')
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('Cluster count $C(t)$')
        ax.set_title(f'Cluster Count Over Time\nfinal={final_n_clusters},  '
                     f'returned to k: {bool(returned_to_k)}')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])
        ax.set_ylim([max(0, k_true - 1), k_true + 3])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches='tight')
        plt.close()

    return {
        'target':            target_key,
        'm':                 m_orig,
        'T_original':        T_original,
        'k_true':            k_true,
        'injection_type':    injection_type,
        'b_inject':          f'{b_inject:.4f}',
        'a_inject_initial':  f'{a_inject_initial:.4f}',
        'a_inject_final':    f'{a_inject_final:.4f}',
        'a_decayed':         a_decayed,
        'b_inject_final':    f'{b_inject_final:.4f}',
        'b_merged':          b_merged,
        'final_n_clusters':  final_n_clusters,
        'returned_to_k':     returned_to_k,
    }

# =============================================================================
# Summary figure
# =============================================================================

def make_summary_figure(results):
    """
    Two panels:
      Left:  table-style heatmap showing returned_to_k for every
             (run, injection_type) combination
      Right: bar chart of overall return rate per injection type
    """
    targets_present = list(dict.fromkeys(r['target'] for r in results))
    inj_types       = ['near', 'isolated']

    # Build matrix: rows = runs (target+m+T), cols = injection type
    run_labels = []
    matrix     = []
    for r in results:
        label = f"{r['target']}  m={r['m']}  T={r['T_original']}"
        if label not in run_labels:
            run_labels.append(label)

    # For each run label, fill in returned_to_k for each injection type
    data = {label: {} for label in run_labels}
    for r in results:
        label = f"{r['target']}  m={r['m']}  T={r['T_original']}"
        data[label][r['injection_type']] = int(r['returned_to_k'])

    mat = np.array([[data[lbl].get(it, -1) for it in inj_types]
                    for lbl in run_labels], dtype=float)
    mat[mat < 0] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(run_labels) * 0.4 + 2)))
    fig.suptitle('Goal 2: Instability of k+1 Cluster Configurations', fontsize=13)

    # Left: heatmap
    ax = axes[0]
    masked = np.ma.masked_invalid(mat)
    cmap   = matplotlib.colors.ListedColormap(['crimson', 'limegreen'])
    im     = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(inj_types)))
    ax.set_xticklabels(inj_types, fontsize=10)
    ax.set_yticks(range(len(run_labels)))
    ax.set_yticklabels(run_labels, fontsize=7)
    ax.set_title('Returned to k clusters?\n(green = yes, red = no)')
    # Annotate cells
    for i in range(len(run_labels)):
        for j in range(len(inj_types)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, 'YES' if val == 1 else 'NO',
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')
    plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.4)

    # Right: bar chart of return rate per injection type
    ax = axes[1]
    for ji, inj in enumerate(inj_types):
        vals = [int(r['returned_to_k']) for r in results
                if r['injection_type'] == inj]
        rate = sum(vals) / len(vals) if vals else 0.0
        ax.bar(ji, rate, color=['steelblue', 'darkorange'][ji],
               edgecolor='black', label=f'{inj}  ({sum(vals)}/{len(vals)})')
    ax.set_xticks(range(len(inj_types)))
    ax.set_xticklabels(inj_types)
    ax.set_ylabel('Fraction returned to k')
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='k', lw=1, linestyle='--', alpha=0.5)
    ax.set_title('Return Rate by Injection Type')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Summary figure -> {SUMMARY_PNG}')

# =============================================================================
# Worker wrapper  (module-level for multiprocessing pickling on Windows)
# =============================================================================

def _worker(args):
    return test_one(args)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()

    # ── Scan all run folders for converged runs (n_clusters == k_true) ─────────
    qualifying = []
    for target_key in os.listdir(FIG_BASE):
        target_path = os.path.join(FIG_BASE, target_key)
        if not os.path.isdir(target_path) or target_key not in TARGETS:
            continue
        for m_dir in os.listdir(target_path):
            m_path = os.path.join(target_path, m_dir)
            if not os.path.isdir(m_path):
                continue
            for t_dir in os.listdir(m_path):
                t_path = os.path.join(m_path, t_dir)
                if not os.path.isdir(t_path):
                    continue
                run_data = load_converged_run(t_path)
                if run_data is not None:
                    qualifying.append((t_path, run_data))

    qualifying.sort(key=lambda x: (x[1]['target'], x[1]['m'], x[1]['T_original']))
    print(f'Found {len(qualifying)} converged runs (n_clusters == k_true)')

    # ── Load existing results for restart safety ───────────────────────────────
    existing = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['target'], int(row['m']),
                       int(row['T_original']), row['injection_type'])
                existing[key] = row
        print(f'Loaded {len(existing)} existing results from {RESULTS_CSV}')

    # ── Build job list ─────────────────────────────────────────────────────────
    injection_types = ['near', 'isolated']
    jobs       = []
    skip_count = 0

    for run_dir, run_data in qualifying:
        for inj_type in injection_types:
            key     = (run_data['target'], run_data['m'],
                       run_data['T_original'], inj_type)
            out_fig = os.path.join(run_dir, f'goal2_{inj_type}.png')
            if key in existing and os.path.exists(out_fig):
                skip_count += 1
                print(f'  SKIP  {run_data["target"]:<12}  '
                      f'm={run_data["m"]:<5}  T={run_data["T_original"]:<6}  '
                      f'{inj_type}')
            else:
                jobs.append((run_dir, run_data, inj_type))

    # Cap workers conservatively: ODE integration for large m is memory-heavy.
    # Each spawned process carries ~100 MB Python runtime overhead on top of
    # the numpy arrays, so 20 workers at m=1500 can exhaust available RAM.
    n_workers = max(1, min(cpu_count() - 2, 6))
    print(f'\nQualifying runs : {len(qualifying)}')
    print(f'Jobs to run     : {len(jobs)}  ({len(injection_types)} injection types each)')
    print(f'Skipped         : {skip_count}')
    print(f'Workers         : {n_workers}')

    # ── Run in parallel ────────────────────────────────────────────────────────
    all_results = dict(existing)
    new_count   = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker, jobs):
            if result is None:
                continue
            key = (result['target'], int(result['m']),
                   int(result['T_original']), result['injection_type'])
            all_results[key] = result
            new_count += 1
            status = 'RETURNED to k' if result['returned_to_k'] else 'STAYED at k+1'
            print(f'  DONE  {result["target"]:<12}  '
                  f'm={result["m"]:<5}  T={result["T_original"]:<6}  '
                  f'{result["injection_type"]:<10}  '
                  f'b_inject={result["b_inject"]}  '
                  f'a_decayed={bool(result["a_decayed"])}  '
                  f'b_merged={bool(result["b_merged"])}  '
                  f'final_C={result["final_n_clusters"]}  '
                  f'[{status}]')

    # ── Write results CSV ──────────────────────────────────────────────────────
    sorted_results = sorted(
        all_results.values(),
        key=lambda r: (r['target'], int(r['m']),
                       int(r['T_original']), r['injection_type'])
    )
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
    print(f'\nDone.  {new_count} new,  {skip_count} skipped,  '
          f'wall time {wall:.1f}s')
