"""
regenerate_figures.py
=====================
Post-processing script that regenerates a clean center-panel figure
for every completed run folder.

The original bias_trajectories.png / slide93_reproduction.png may plot all m
bias dots which is cluttered for large m. This script creates a new figure
'final_fit_clean.png' in each run folder that shows only the cluster center
locations as vertical tick marks, making the figure readable.

This script only needs convergence_check.csv and run_meta.csv, which are
written for every completed run.

Mode switch
-----------
  Set MODE = "flow"     → reads from figures/Replication data/  (ODE results)
  Set MODE = "discrete" → reads from figures/Discrete GD/        (GD results)

Run after the relevant simulation script:
    python regenerate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, csv

# =============================================================================
# ── Mode switch ───────────────────────────────────────────────────────────────
# "flow"     → ODE results   in figures/Replication data/
# "discrete" → GD results    in figures/Discrete GD/
# =============================================================================
MODE = "flow"
# "flow"     → ODE results in figures/Replication data/
# "discrete" → GD results  in figures/Discrete GD/

if MODE == "flow":
    FIG_BASE   = os.path.join('figures', 'Replication data')
    TIME_FIELD = 'T'
    TIME_LABEL = 'T'
else:
    FIG_BASE   = os.path.join('figures', 'Discrete GD')
    TIME_FIELD = 'steps'
    TIME_LABEL = 'steps'

# =============================================================================
# Constants — must match the simulation scripts
# =============================================================================
N_QUAD      = 400
X_QUAD      = np.linspace(-1.0, 1.0, N_QUAD)
CLUSTER_TOL = 0.02

# =============================================================================
# Target functions
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

def eval_network(x, a, b):
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def get_cluster_centers(b, tol=CLUSTER_TOL):
    """Return centroid of each cluster."""
    sorted_b = np.sort(b)
    centers, group = [], [sorted_b[0]]
    for i in range(1, len(sorted_b)):
        if sorted_b[i] - sorted_b[i - 1] > tol:
            centers.append(float(np.mean(group)))
            group = []
        group.append(sorted_b[i])
    centers.append(float(np.mean(group)))
    return np.array(centers)

# =============================================================================
# Load one run folder
# =============================================================================

def load_run(run_dir):
    f_csv  = os.path.join(run_dir, 'convergence_check.csv')
    f_meta = os.path.join(run_dir, 'run_meta.csv')
    if not (os.path.exists(f_csv) and os.path.exists(f_meta)):
        return None
    with open(f_meta, newline='') as mf:
        meta = next(csv.DictReader(mf))
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
# Generate the clean figure
# =============================================================================

def make_clean_figure(run_dir, run_data):
    """
    Saves 'final_fit_clean.png' in run_dir.

    Center panel (only): target f* vs network f, with cluster centers shown
    as vertical tick marks at y=0 instead of all m bias dots.
    """
    out_path = os.path.join(run_dir, 'final_fit_clean.png')
    if os.path.exists(out_path):
        return False   # already done

    target_key   = run_data['target']
    m            = run_data['m']
    T            = run_data['T']
    k_true       = run_data['k_true']
    n_clusters   = run_data['n_clusters']
    b            = run_data['b']
    a            = run_data['a']

    if target_key not in TARGETS:
        return False
    target_label, _, f_star = TARGETS[target_key]

    # Compute quantities
    x_plot          = np.linspace(-1.0, 1.0, 500)
    f_final         = eval_network(x_plot, a, b)
    cluster_centers = get_cluster_centers(b)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(
        f'target = {target_label},   $m = {m}$,   ${TIME_LABEL} = {T}$\n'
        f'clusters = {n_clusters},   $k = {k_true}$,   '
        f'$C = k$: {n_clusters == k_true}',
        fontsize=11)

    # Target and network
    ax.plot(x_plot, f_star(x_plot), 'k--', lw=2, label=f'Target {target_label}')
    ax.plot(x_plot, f_final, color='crimson', lw=2, label='Network $f$')

    # Cluster centers as vertical tick marks at y=0
    y_min, y_max = ax.get_ylim()
    tick_height  = (y_max - y_min) * 0.06   # 6% of plot height
    ax.vlines(cluster_centers,
              ymin=-tick_height / 2,
              ymax= tick_height / 2,
              colors='steelblue', linewidths=2.5, zorder=5)

    # Legend entry for cluster centers
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='steelblue', lw=2.5,
                          label=f'Cluster centers ({n_clusters})'))
    ax.legend(handles=handles, fontsize=9, loc='best')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_xlim([-1, 1])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    return True

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    done = 0
    skipped = 0
    print(f'Mode: {MODE}  →  {FIG_BASE}')

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
                run_data = load_run(t_path)
                if run_data is None:
                    continue
                created = make_clean_figure(t_path, run_data)
                if created:
                    done += 1
                    print(f'  DONE  {run_data["target"]:<12}  '
                          f'm={run_data["m"]:<5}  {TIME_LABEL}={run_data["T"]}')
                else:
                    skipped += 1

    print(f'\nDone. {done} figures created, {skipped} already existed.')
