"""
adaptive_cluster_analysis.py

Re-analyzes all simulate_parallel.py runs using an adaptive gap threshold
that scales with neuron density, instead of the fixed 0.02 threshold.

Adaptive threshold = MULTIPLIER * mean inter-neuron spacing of active neurons.
Several multiplier values are tested; PRIMARY_MULT is used for all per-run plots.

Outputs to adaptive_cluster_plots/ — never overwrites existing files.

Plots produced:
  Global:
    adaptive_convergence_plot.png      C(m) vs m for all 9 targets, adaptive vs fixed
    multiplier_comparison.png          C(m) for each tested multiplier, all targets

  Per run (only for runs where adaptive C differs from fixed C, or always for dagger runs):
    {target}_m={m}_final_fit_adaptive.png        Final fit + adaptive cluster ticks
    {target}_m={m}_clusters_vs_inflections_adaptive.png  Cluster vs inflection alignment
    {target}_m={m}_ode_verification_adaptive.png  da/dt and R_j scatter (same data, new labels)
    {target}_m={m}_bias_density_adaptive.png      Histogram of bias positions with cluster marks
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT    = Path("figures/Replication data")
OUT_DIR      = Path("adaptive_cluster_plots")
OUT_DIR.mkdir(exist_ok=True)

ACTIVE_THRESHOLD = 0.05
FIXED_THRESHOLD  = 0.02
MULTIPLIERS      = [3, 5, 10, 20]
PRIMARY_MULT     = 5

M_VALUES = [50, 100, 250, 500, 1000, 1500, 2000, 3500, 5000]
T = 500

# ── Target definitions ─────────────────────────────────────────────────────────
def inflections_sin(n):
    return [j / n for j in range(-(n - 1), n) if abs(j / n) < 1]

TARGET_INFO = {
    "sin_1pi": {
        "k": 1, "label": r"$\sin(\pi x)$",
        "func": lambda x: np.sin(np.pi * x),
        "inflections": [0.0],
    },
    "sin_2pi": {
        "k": 3, "label": r"$\sin(2\pi x)$",
        "func": lambda x: np.sin(2 * np.pi * x),
        "inflections": [-0.5, 0.0, 0.5],
    },
    "sin_3pi": {
        "k": 5, "label": r"$\sin(3\pi x)$",
        "func": lambda x: np.sin(3 * np.pi * x),
        "inflections": inflections_sin(3),
    },
    "sin_4pi": {
        "k": 7, "label": r"$\sin(4\pi x)$",
        "func": lambda x: np.sin(4 * np.pi * x),
        "inflections": inflections_sin(4),
    },
    "sin_5pi": {
        "k": 9, "label": r"$\sin(5\pi x)$",
        "func": lambda x: np.sin(5 * np.pi * x),
        "inflections": inflections_sin(5),
    },
    "sin_6pi": {
        "k": 11, "label": r"$\sin(6\pi x)$",
        "func": lambda x: np.sin(6 * np.pi * x),
        "inflections": inflections_sin(6),
    },
    "sin_7pi": {
        "k": 13, "label": r"$\sin(7\pi x)$",
        "func": lambda x: np.sin(7 * np.pi * x),
        "inflections": inflections_sin(7),
    },
    "poly_k3": {
        "k": 3, "label": r"$x^5 - 3x^3$",
        "func": lambda x: x**5 - 3 * x**3,
        "inflections": [-np.sqrt(9 / 10), 0.0, np.sqrt(9 / 10)],
    },
    "x_cubed": {
        "k": 1, "label": r"$x^3$",
        "func": lambda x: x**3,
        "inflections": [0.0],
    },
}

TARGET_ORDER = [
    "sin_1pi", "sin_2pi", "sin_3pi", "sin_4pi",
    "sin_5pi", "sin_6pi", "sin_7pi", "poly_k3", "x_cubed",
]

# ── Clustering helpers ─────────────────────────────────────────────────────────
def count_clusters_adaptive(b_active, multiplier):
    if len(b_active) < 2:
        return len(b_active)
    b = np.sort(b_active)
    mean_spacing = (b[-1] - b[0]) / (len(b) - 1)
    threshold = multiplier * mean_spacing
    return int((np.diff(b) > threshold).sum() + 1)

def count_clusters_fixed(b_active, threshold=FIXED_THRESHOLD):
    if len(b_active) < 2:
        return len(b_active)
    return int((np.diff(np.sort(b_active)) > threshold).sum() + 1)

def get_cluster_centers(b_active, multiplier):
    if len(b_active) == 0:
        return np.array([])
    b = np.sort(b_active)
    if len(b) == 1:
        return b
    mean_spacing = (b[-1] - b[0]) / (len(b) - 1)
    threshold = multiplier * mean_spacing
    split_pts = np.where(np.diff(b) > threshold)[0] + 1
    clusters = np.split(b, split_pts)
    return np.array([c.mean() for c in clusters])

def network_output(x, a, b):
    """f(x) = sum_j a_j * relu(x - b_j)"""
    return np.array([np.sum(a * np.maximum(xi - b, 0.0)) for xi in x])

# ── Load all runs ──────────────────────────────────────────────────────────────
print("Loading all runs...")
records = []

for target in TARGET_ORDER:
    info = TARGET_INFO[target]
    for m in M_VALUES:
        run_dir = DATA_ROOT / target / f"m={m}" / f"T={T}"
        csv_path = run_dir / "convergence_check.csv"
        meta_path = run_dir / "run_meta.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        active_mask = df["is_active"] == 1
        b_active = df.loc[active_mask, "b_j"].values
        a_active = df.loc[active_mask, "a_j"].values
        n_active = int(active_mask.sum())

        meta = pd.read_csv(meta_path).iloc[0]
        c_fixed = int(meta["n_clusters"])
        loss    = float(meta["loss"])
        max_da  = float(meta["max_da"])

        row = {
            "target": target, "m": m, "k": info["k"],
            "n_active": n_active,
            "c_fixed": c_fixed,
            "loss": loss, "max_da": max_da,
        }
        for mult in MULTIPLIERS:
            row[f"c_adapt_{mult}"] = count_clusters_adaptive(b_active, mult)
        records.append(row)

summary = pd.DataFrame(records)
out_csv = OUT_DIR / "adaptive_threshold_summary.csv"
summary.to_csv(out_csv, index=False)
print(f"Summary saved to {out_csv}")

# Print comparison table
cols = ["target", "m", "k", "n_active", "c_fixed"] + [f"c_adapt_{mult}" for mult in MULTIPLIERS]
print("\nC values: fixed vs adaptive (multipliers: {})".format(MULTIPLIERS))
print(summary[cols].to_string(index=False))

# ── Global: adaptive convergence plot ─────────────────────────────────────────
print("\nGenerating adaptive convergence plot...")

n_targets = len(TARGET_ORDER)
ncols = 3
nrows = (n_targets + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()

for idx, target in enumerate(TARGET_ORDER):
    ax = axes[idx]
    info = TARGET_INFO[target]
    sub = summary[summary["target"] == target].sort_values("m")

    ax.plot(sub["m"], sub["c_fixed"],
            "o--", color="gray", lw=1.5, ms=5, label="Fixed (0.02)")
    ax.plot(sub["m"], sub[f"c_adapt_{PRIMARY_MULT}"],
            "s-", color="#1f77b4", lw=2, ms=6, label=f"Adaptive ({PRIMARY_MULT}×)")
    ax.axhline(info["k"], color="red", lw=1.5, ls=":", label=f"k={info['k']}")

    ax.set_title(info["label"], fontsize=12)
    ax.set_xlabel("m")
    ax.set_ylabel("C(m)")
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

for idx in range(n_targets, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f"C(m) vs m: Fixed threshold (0.02) vs Adaptive threshold ({PRIMARY_MULT}× mean spacing)",
    fontsize=13, y=1.01
)
fig.tight_layout()
out_path = OUT_DIR / "adaptive_convergence_plot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_path}")

# ── Global: multiplier comparison ─────────────────────────────────────────────
print("Generating multiplier comparison plot...")

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = axes.flatten()
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

for idx, target in enumerate(TARGET_ORDER):
    ax = axes[idx]
    info = TARGET_INFO[target]
    sub = summary[summary["target"] == target].sort_values("m")

    ax.plot(sub["m"], sub["c_fixed"],
            "o--", color="gray", lw=1.5, ms=4, alpha=0.6, label="Fixed 0.02")
    for mult, col in zip(MULTIPLIERS, colors):
        ax.plot(sub["m"], sub[f"c_adapt_{mult}"],
                "s-", color=col, lw=1.5, ms=4, label=f"{mult}×")
    ax.axhline(info["k"], color="black", lw=1.5, ls=":", label=f"k={info['k']}")

    ax.set_title(info["label"], fontsize=12)
    ax.set_xlabel("m")
    ax.set_ylabel("C(m)")
    ax.legend(fontsize=6)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

for idx in range(n_targets, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    "C(m) vs m: Fixed threshold vs Adaptive (multipliers: {})".format(MULTIPLIERS),
    fontsize=13, y=1.01
)
fig.tight_layout()
out_path = OUT_DIR / "multiplier_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_path}")

# ── Per-run plots ──────────────────────────────────────────────────────────────
# Generate for every run where adaptive C (primary mult) != fixed C,
# plus all runs where fixed C = 1 and k > 1 (the dagger runs).
x_plot = np.linspace(-1, 1, 500)

print("\nGenerating per-run plots...")
for _, row in summary.iterrows():
    target = row["target"]
    m      = int(row["m"])
    k      = int(row["k"])
    c_fix  = int(row["c_fixed"])
    c_adap = int(row[f"c_adapt_{PRIMARY_MULT}"])

    is_dagger = (c_fix == 1 and k > 1)
    changed   = (c_adap != c_fix)
    if not (is_dagger or changed):
        continue

    info    = TARGET_INFO[target]
    run_dir = DATA_ROOT / target / f"m={m}" / f"T={T}"
    df      = pd.read_csv(run_dir / "convergence_check.csv")

    active      = df[df["is_active"] == 1].copy()
    b_all       = df["b_j"].values
    a_all       = df["a_j"].values
    b_act       = active["b_j"].values
    a_act       = active["a_j"].values
    da_act      = active["da_dt"].values
    Rj_act      = active["R_j"].values
    b_inact     = df[df["is_active"] == 0]["b_j"].values

    centers_adap = get_cluster_centers(b_act, PRIMARY_MULT)
    n_active     = len(b_act)
    mean_spacing = (b_act.max() - b_act.min()) / max(n_active - 1, 1) if n_active > 1 else 0
    threshold    = PRIMARY_MULT * mean_spacing

    tag = f"{target}_m={m}"

    # ── 1. Final fit + adaptive cluster ticks ─────────────────────────────────
    out_path = OUT_DIR / f"{tag}_final_fit_adaptive.png"
    if not out_path.exists():
        fstar = info["func"](x_plot)
        f_net = network_output(x_plot, a_all, b_all)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_plot, fstar, "k-",  lw=1.5, label=r"$f^*$")
        ax.plot(x_plot, f_net, "b--", lw=1.5, label=r"$f$ (network)", alpha=0.8)

        y_min = min(fstar.min(), f_net.min()) - 0.15
        tick_y = y_min - 0.05

        ax.plot(centers_adap, np.full_like(centers_adap, tick_y),
                "|", color="#1f77b4", ms=14, mew=2.5,
                label=f"Adaptive clusters (C={len(centers_adap)})")
        for xi in info["inflections"]:
            ax.axvline(xi, color="red", lw=0.8, ls=":", alpha=0.5)

        ax.set_title(
            f"{info['label']},  m={m},  k={k}\n"
            f"Fixed C={c_fix}  →  Adaptive C={c_adap}  (threshold={threshold:.4f})",
            fontsize=10
        )
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")

    # ── 2. Clusters vs inflection points ──────────────────────────────────────
    out_path = OUT_DIR / f"{tag}_clusters_vs_inflections_adaptive.png"
    if not out_path.exists():
        inflections = np.array(info["inflections"])
        fig, ax = plt.subplots(figsize=(8, 2.5))

        ax.scatter(centers_adap, np.ones(len(centers_adap)),
                   marker="|", s=300, lw=2.5, color="#1f77b4",
                   label=f"Adaptive cluster centers (C={len(centers_adap)})")
        ax.scatter(inflections, np.zeros(len(inflections)),
                   marker="|", s=300, lw=2.5, color="red",
                   label=f"Inflection points (k={k})")

        ax.set_xlim(-1.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Inflections", "Clusters"], fontsize=10)
        ax.set_xlabel("x")
        ax.set_title(
            f"{info['label']},  m={m},  k={k}\n"
            f"Fixed C={c_fix}  →  Adaptive C={c_adap}",
            fontsize=10
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")

    # ── 3. ODE verification (da/dt and R_j vs b_j) ────────────────────────────
    out_path = OUT_DIR / f"{tag}_ode_verification_adaptive.png"
    if not out_path.exists():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.scatter(b_act, da_act, s=3, alpha=0.5, color="#1f77b4")
        ax1.axhline(0, color="k", lw=1)
        for c in centers_adap:
            ax1.axvline(c, color="red", lw=0.8, ls="--", alpha=0.6)
        ax1.set_xlabel(r"$b_j$")
        ax1.set_ylabel(r"$\dot{a}_j$")
        ax1.set_title(f"ODE velocity $\\dot{{a}}_j$\n(red = adaptive cluster centers)")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(b_act, Rj_act, s=3, alpha=0.5, color="#ff7f0e")
        ax2.axhline(0, color="k", lw=1)
        for c in centers_adap:
            ax2.axvline(c, color="red", lw=0.8, ls="--", alpha=0.6)
        ax2.set_xlabel(r"$b_j$")
        ax2.set_ylabel(r"$R_j$")
        ax2.set_title(r"Fixed-point residual $R_j = \int_{b_j}^{1}(f - f^*)\,dx$")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"{info['label']},  m={m},  k={k},  "
            f"Fixed C={c_fix}  →  Adaptive C={c_adap}",
            fontsize=10
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")

    # ── 4. Bias density histogram ──────────────────────────────────────────────
    out_path = OUT_DIR / f"{tag}_bias_density_adaptive.png"
    if not out_path.exists():
        n_bins = max(30, n_active // 20)
        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.hist(b_act, bins=n_bins, color="#1f77b4", alpha=0.7, edgecolor="white", lw=0.3)
        for c in centers_adap:
            ax.axvline(c, color="#1f77b4", lw=2, ls="--", alpha=0.9)
        for xi in info["inflections"]:
            ax.axvline(xi, color="red", lw=1.5, ls=":", alpha=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#1f77b4", lw=2, ls="--",
                   label=f"Adaptive cluster centers (C={len(centers_adap)})"),
            Line2D([0], [0], color="red", lw=1.5, ls=":",
                   label=f"Inflection points (k={k})"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        ax.set_xlabel(r"$b_j$ (active neurons)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{info['label']},  m={m},  n_active={n_active}\n"
            f"Adaptive threshold = {PRIMARY_MULT}× mean spacing = {threshold:.5f}   "
            f"Fixed C={c_fix}  →  Adaptive C={c_adap}",
            fontsize=9
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")

# ── Summary table: where does adaptive C = k? ─────────────────────────────────
print("\n--- Cases where adaptive C matches k (primary mult={}) ---".format(PRIMARY_MULT))
match = summary[summary[f"c_adapt_{PRIMARY_MULT}"] == summary["k"]]
print(match[["target", "m", "k", "c_fixed", f"c_adapt_{PRIMARY_MULT}"]].to_string(index=False))

print("\n--- Cases where adaptive C changed from fixed C ---")
changed = summary[summary[f"c_adapt_{PRIMARY_MULT}"] != summary["c_fixed"]]
print(changed[["target", "m", "k", "c_fixed", f"c_adapt_{PRIMARY_MULT}"]].to_string(index=False))

print(f"\nDone. All outputs in {OUT_DIR}/")
