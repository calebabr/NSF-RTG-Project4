"""
plot_convergence_now.py
=======================
Generates a convergence plot from whatever run_meta.csv files currently
exist on disk — works at any point during or after a simulation script.

Scans every run folder under the selected figures directory, reads run_meta.csv
where available, and produces a convergence plot showing C(m, f*) vs m.

Safe to run while a simulation is still running. Already-completed runs have
their run_meta.csv written and will be included.

Mode switch
-----------
  Set MODE = "flow"     → reads from figures/Replication data/  (ODE results)
  Set MODE = "discrete" → reads from figures/Discrete GD/        (GD results)

Output
------
  {FIG_BASE}/convergence_plot_current.png

Usage
-----
    python plot_convergence_now.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, tempfile

# =============================================================================
# ── Mode switch ───────────────────────────────────────────────────────────────
# "flow"     → ODE results   in figures/Replication data/
# "discrete" → GD results    in figures/Discrete GD/
# =============================================================================
MODE = "flow"
# "flow"     → ODE results in figures/Replication data/
# "discrete" → GD results  in figures/Discrete GD/

if MODE == "flow":
    FIG_BASE         = os.path.join('figures', 'Replication data')
    TIME_FIELD       = 'T'
    TIME_LABEL       = 'T'
    INNER_DIR_PREFIX = 'T='
    ONLY_T           = 500    # only include runs with exactly T=500
else:
    FIG_BASE         = os.path.join('figures', 'Discrete GD')
    TIME_FIELD       = 'steps'
    TIME_LABEL       = 'steps'
    INNER_DIR_PREFIX = 'steps='
    ONLY_T           = None   # no T filter in discrete mode

OUT_PATH = os.path.join(FIG_BASE, 'convergence_plot_current.png')

# Analytical k values — used for the horizontal reference line per target
K_TRUE = {
    'sin_1pi': 1,
    'x_cubed': 1,
    'sin_2pi': 3,
    'poly_k3': 3,
    'sin_3pi': 5,
    'sin_4pi': 7,
    'sin_5pi': 9,
    'sin_6pi': 11,
    'sin_7pi': 13,
}

LABELS = {
    'sin_1pi': r'$\sin(\pi x)$',
    'x_cubed': r'$x^3$',
    'sin_2pi': r'$\sin(2\pi x)$',
    'poly_k3': r'$x^5 - 3x^3$',
    'sin_3pi': r'$\sin(3\pi x)$',
    'sin_4pi': r'$\sin(4\pi x)$',
    'sin_5pi': r'$\sin(5\pi x)$',
    'sin_6pi': r'$\sin(6\pi x)$',
    'sin_7pi': r'$\sin(7\pi x)$',
}

# =============================================================================
# Collect all available run_meta.csv data
# =============================================================================

rows = []

for target_key in os.listdir(FIG_BASE):
    target_path = os.path.join(FIG_BASE, target_key)
    if not os.path.isdir(target_path) or target_key not in K_TRUE:
        continue
    for m_dir in os.listdir(target_path):
        m_path = os.path.join(target_path, m_dir)
        if not os.path.isdir(m_path) or not m_dir.startswith('m='):
            continue
        for t_dir in os.listdir(m_path):
            t_path = os.path.join(m_path, t_dir)
            if not os.path.isdir(t_path) or not t_dir.startswith(INNER_DIR_PREFIX):
                continue
            meta_file = os.path.join(t_path, 'run_meta.csv')
            if not os.path.exists(meta_file):
                continue
            with open(meta_file, newline='') as f:
                row = next(csv.DictReader(f))
            T_val = int(row[TIME_FIELD])
            if MODE == 'flow' and T_val != ONLY_T:
                continue   # only include T=500 runs
            rows.append({
                'target':     target_key,
                'm':          int(row['m']),
                'T':          T_val,
                'k_true':     int(row['k_true']),
                'n_clusters': int(row['n_clusters']),
            })

print(f'Found {len(rows)} completed runs across '
      f'{len(set(r["target"] for r in rows))} targets.')

if not rows:
    print('No run_meta.csv files found. Nothing to plot.')
    exit()

# =============================================================================
# Build the convergence plot
# =============================================================================

targets_present = list(dict.fromkeys(
    k for k in LABELS if any(r['target'] == k for r in rows)
))

# In flow mode, group series by T so multiple ODE runs appear as separate lines.
# In discrete mode, steps vary with m (scaled lr), so group everything as one
# series per target — splitting by T would give one isolated dot per m.
if MODE == 'flow':
    T_values_present = sorted(set(r['T'] for r in rows))
else:
    T_values_present = [None]   # sentinel: plot all m as a single connected line

n_targets = len(targets_present)
ncols     = 3
nrows     = -(-n_targets // ncols)   # ceiling division

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(6 * ncols, 5 * nrows),
                          squeeze=False)
axes_flat = axes.flatten()

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*']
cmap    = plt.cm.tab10(np.linspace(0, 0.85, max(len(T_values_present), 1)))

for ax_i, tkey in enumerate(targets_present):
    ax     = axes_flat[ax_i]
    k_true = K_TRUE[tkey]
    label  = LABELS[tkey]

    for ti, T in enumerate(T_values_present):
        if T is None:
            # Discrete mode: one series across all m, take best (highest-steps) run per m
            best = {}
            for r in rows:
                if r['target'] != tkey:
                    continue
                if r['m'] not in best or r['T'] > best[r['m']]['T']:
                    best[r['m']] = r
            pts       = sorted(best.values(), key=lambda r: r['m'])
            series_lbl = 'Discrete GD (scaled lr)'
        else:
            pts = sorted(
                [r for r in rows if r['target'] == tkey and r['T'] == T],
                key=lambda r: r['m']
            )
            series_lbl = f'{TIME_LABEL}={T}'

        if not pts:
            continue
        ms = [r['m']          for r in pts]
        cs = [r['n_clusters'] for r in pts]
        ax.plot(ms, cs,
                marker=markers[ti % len(markers)],
                color=cmap[ti],
                lw=1.5, ms=5,
                label=series_lbl)

    # Horizontal line at analytical k
    ax.axhline(k_true, color='crimson', lw=2, linestyle='--',
               label=f'k = {k_true}')

    ax.set_xlabel('Width $m$', fontsize=10)
    ax.set_ylabel('Cluster count $C(m, f^*)$', fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

# Hide unused axes
for ax_i in range(len(targets_present), len(axes_flat)):
    axes_flat[ax_i].set_visible(False)

n_done    = len(rows)
n_targets = len(set(r['target'] for r in rows))
if MODE == 'flow':
    time_note = f'{TIME_LABEL} = {ONLY_T}'
else:
    step_vals = sorted(set(r['T'] for r in rows))
    time_note = f'steps = {step_vals[0]}–{step_vals[-1]} (scaled with m)'
fig.suptitle(
    f'Open Problem 4.1 — $C(m,f^*) \\to k$ as $m \\to \\infty$  [{MODE} mode]\n'
    f'Crimson dashed = analytical $k$    '
    f'({n_done} completed runs across {n_targets} targets  |  {time_note})',
    fontsize=13)

plt.tight_layout()

# Save to a temp file in the same directory, then atomically replace the
# target so a locked/open PNG (image viewer, OneDrive sync) can't block us.
_dir = os.path.dirname(OUT_PATH)
_fd, _tmp = tempfile.mkstemp(suffix='.png', dir=_dir)
os.close(_fd)
try:
    plt.savefig(_tmp, bbox_inches='tight', dpi=130)
    plt.close()
    os.replace(_tmp, OUT_PATH)
except Exception:
    plt.close()
    try:
        os.remove(_tmp)
    except OSError:
        pass
    raise

print(f'Saved -> {OUT_PATH}')
