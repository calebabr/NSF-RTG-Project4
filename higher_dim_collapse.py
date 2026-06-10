"""
Open Problem 4.2 — Parameter Collapse in d=2  (v2)
===================================================
Changes from v1:
  1. Ridge uses a non-axis-aligned u = (3/5, 4/5) so collapse toward
     an oblique angle is distinguishable from the separable target.
  2. Broken-symmetry separable: sin(2πx1) + 0.3 sin(2πx2) so the two
     direction families have unequal weight and can be told apart.
  3. Width sweep: run each target at m = 64, 256, 512 to test whether
     collapse is a large-width phenomenon.
  4. Longer training: 6000 epochs, snapshotting every 300.
  5. Fixed pruning diagnostic: skip merges where neurons have opposite-
     sign alpha (they form a cancelling pair, not a redundant pair).
     Also retrain the pruned network for 200 steps before evaluating loss.

Run:
    pip install torch scikit-learn scipy matplotlib numpy
    python collapse_v2.py

Outputs:
    collapse_v2_results.json   — slim diagnostics (no point clouds)
    collapse_v2_plots/         — per-target, per-width plots + summary
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import DBSCAN
from scipy.stats import entropy as scipy_entropy
import json, os, itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────────────

WIDTHS         = [64, 256, 512, 1024]
N_TRAIN        = 4096
N_TEST         = 1024
N_EPOCHS       = 6000
LR             = 1e-3
SNAPSHOT_EVERY = 300
DBSCAN_EPS     = 0.12
PRUNE_EPS      = 0.10
PRUNE_RETRAIN  = 200      # steps to retrain pruned net before eval
N_BINS         = 36

# Non-axis-aligned ridge direction
U_RIDGE = torch.tensor([3/5, 4/5], dtype=torch.float32)

# ─── Model ────────────────────────────────────────────────────────────────────

class ShallowReLU(nn.Module):
    """f(x) = sum_j a_j relu(w_j^T x + b_j),  x in R^2."""
    def __init__(self, m):
        super().__init__()
        self.W = nn.Parameter(torch.randn(m, 2) * 0.5)
        self.b = nn.Parameter(torch.randn(m) * 0.5)
        self.a = nn.Parameter(torch.randn(m) * 0.1)

    def forward(self, x):
        return (torch.relu(x @ self.W.T + self.b) * self.a).sum(1)

    def canonical(self):
        """
        Reparameterize into scale-invariant coordinates:
            theta_j = angle of w_j         (which direction the neuron is sensitive to)
            beta_j  = b_j / ||w_j||        (signed distance from origin to kink)
            alpha_j = a_j * ||w_j||        (scale-absorbed output weight)
        Two neurons defining the same hyperplane have identical (theta, beta)
        regardless of how their raw weights are scaled.
        """
        with torch.no_grad():
            norms = self.W.norm(dim=1).clamp(1e-8)
            n     = self.W / norms.unsqueeze(1)
            beta  = self.b / norms
            alpha = self.a * norms
            theta = torch.remainder(
                torch.atan2(n[:, 1], n[:, 0]),
                np.pi
            )
        return theta.numpy(), beta.numpy(), alpha.numpy()

# ─── Targets ──────────────────────────────────────────────────────────────────

def ridge(x):
    """
    sin(2π u^T x) + 0.5 sin(4π u^T x),  u = (3/5, 4/5)
    Varies only along u ≈ 53°. All normals should collapse toward θ ≈ 53°.
    Using oblique u breaks the axis-alignment confound with separable.
    """
    proj = x @ U_RIDGE
    return torch.sin(2 * np.pi * proj) + 0.5 * torch.sin(4 * np.pi * proj)

def separable(x):
    """
    sin(2π x1) + 0.3 sin(2π x2)
    Broken symmetry: x1-term is 3x stronger, so the two direction families
    should have unequal sizes. Cleaner test of family splitting than equal weights.
    Expected: larger cluster at θ ≈ 0° (x1), smaller at θ ≈ 90° (x2).
    """
    return torch.sin(2 * np.pi * x[:, 0]) + 0.3 * torch.sin(2 * np.pi * x[:, 1])

def radial(x):
    """
    cos(π ||x||)
    Rotationally symmetric — no privileged direction.
    Control case: entropy should stay high, no clustering.
    """
    return torch.cos(np.pi * x.norm(dim=1))

TARGETS = {
    "ridge":     (ridge,     "Ridge (u=53°): sin(2πuᵀx)+0.5sin(4πuᵀx)"),
    "separable": (separable, "Separable (asymmetric): sin(2πx₁)+0.3sin(2πx₂)"),
    "radial":    (radial,    "Radial: cos(π‖x‖)"),
}

# ─── Diagnostics ──────────────────────────────────────────────────────────────

def angular_entropy(theta, alpha):
    """
    Shannon entropy of binned angular distribution.
    H=0: all neurons same direction (full collapse).
    H≈3.58: uniform (no collapse).
    """
    weights = np.abs(alpha)

    counts, _ = np.histogram(
    theta,
    bins=N_BINS,
    range=(0, np.pi),
    weights=weights
    )

    p = counts.astype(float) + 1e-10
    p /= p.sum()

    return float(-np.sum(p * np.log(p)))

def effective_neuron_count(alpha):
    w = np.abs(alpha)

    if np.sum(w) < 1e-12:
        return 0.0

    return float((np.sum(w) ** 2) / np.sum(w ** 2))

def offset_clustering(theta, beta):
    """DBSCAN on normalized (theta, beta). Returns (n_clusters, labels)."""
    pts = np.stack([theta, beta], axis=1)
    pts = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-8)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=2).fit_predict(pts)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    return int(n), labels

def effective_width(model, x_train, y_train, x_test, y_test):
    """
    Fixed pruning diagnostic (v2 changes):
      - Skip merges where neurons have opposite-sign alpha (cancelling pairs,
        not redundant pairs).
      - Retrain pruned network for PRUNE_RETRAIN steps before evaluating loss.
    Returns (orig_loss, pruned_loss, orig_width, pruned_width).
    """
    theta, beta, alpha = model.canonical()
    pts      = np.stack([theta, beta], axis=1)
    pts_norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-8)

    used     = np.zeros(len(theta), dtype=bool)
    clusters = []
    for i in range(len(theta)):
        if used[i]:
            continue
        dists = np.linalg.norm(pts_norm - pts_norm[i], axis=1)
        nearby = np.where(dists < PRUNE_EPS)[0]

        # --- v2 fix: only merge neurons with the same sign of alpha ---
        sign_i     = np.sign(alpha[i])
        same_sign  = nearby[np.sign(alpha[nearby]) == sign_i]
        cluster    = same_sign if len(same_sign) > 0 else np.array([i])
        used[cluster] = True
        clusters.append(cluster)

    with torch.no_grad():
        orig_loss = float(nn.MSELoss()(model(x_test), y_test))

    # build pruned model
    pruned = ShallowReLU(m=len(clusters))
    with torch.no_grad():
        for new_idx, cluster in enumerate(clusters):
            rep      = cluster[0]
            rep_norm = model.W[rep].norm().clamp(1e-8)
            pruned.W[new_idx] = model.W[rep]
            pruned.b[new_idx] = model.b[rep]
            norms_c  = model.W[cluster].norm(dim=1).clamp(1e-8)
            alphas_c = model.a[cluster] * norms_c
            pruned.a[new_idx] = alphas_c.sum() / rep_norm

    # --- v2 fix: retrain pruned net briefly before evaluating ---
    opt = optim.Adam(pruned.parameters(), lr=LR)
    for _ in range(PRUNE_RETRAIN):
        opt.zero_grad()
        nn.MSELoss()(pruned(x_train), y_train).backward()
        opt.step()

    with torch.no_grad():
        pruned_loss = float(nn.MSELoss()(pruned(x_test), y_test))

    return orig_loss, pruned_loss, len(theta), len(clusters)

# ─── Training ─────────────────────────────────────────────────────────────────

def train(target_fn, target_name, m):
    model     = ShallowReLU(m=m)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    x_train = torch.FloatTensor(N_TRAIN, 2).uniform_(-1, 1)
    y_train = target_fn(x_train)
    x_test  = torch.FloatTensor(N_TEST,  2).uniform_(-1, 1)
    y_test  = target_fn(x_test)

    snapshots = []

    for epoch in range(N_EPOCHS + 1):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        if epoch % SNAPSHOT_EVERY == 0:
            theta, beta, alpha = model.canonical()
            ang_ent = angular_entropy(theta, alpha)
            eff_neurons = effective_neuron_count(alpha)
            active = np.abs(alpha) > 0.01 * np.max(np.abs(alpha))
            n_clusters, labels = offset_clustering(
                theta[active],
                beta[active]
            )
            orig_loss, pruned_loss, orig_w, pruned_w = effective_width(
                model, x_train, y_train, x_test, y_test)

            snapshots.append({
                "epoch":           epoch,
                "train_loss":      float(loss.item()),
                "angular_entropy": ang_ent,
                "n_clusters":      n_clusters,
                "orig_loss":       orig_loss,
                "pruned_loss":     pruned_loss,
                "prune_ratio":     pruned_loss / max(orig_loss, 1e-10),
                "orig_width":      orig_w,
                "pruned_width":    pruned_w,
                "theta":           theta.tolist(),
                "beta":            beta.tolist(),
                "alpha":           alpha.tolist(),
                "effective_neurons": eff_neurons,
            })

            print(
                f"  [{target_name} m={m}] epoch {epoch:5d} | "
                f"loss={loss.item():.4f} | "
                f"H={ang_ent:.3f} | "
                f"eff={eff_neurons:.1f} | "
                f"clusters={n_clusters} | "
                f"pruned={pruned_w}/{orig_w}"
            )

    return snapshots

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_target_width(name, label, snaps_by_width, out_dir):
    """
    For one target: 4 panels showing how each diagnostic evolves,
    with one line per width (64 / 256 / 512).
    """
    os.makedirs(out_dir, exist_ok=True)
    colors = {64: "#2563eb", 256: "#16a34a", 512: "#dc2626", 1024: "#000000"}

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Parameter Collapse — {label}", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    titles = ["Training Loss (log)", "Angular Entropy H({θⱼ})",
              "DBSCAN Cluster Count", "Prune Ratio (pruned/full loss)"]
    keys   = ["train_loss", "angular_entropy", "n_clusters", "prune_ratio"]
    ylogs  = [True, False, False, False]

    for m, snaps in snaps_by_width.items():
        epochs = [s["epoch"] for s in snaps]
        for ax, key, title, ylog in zip(axes, keys, titles, ylogs):
            vals = [s[key] for s in snaps]
            ax.plot(epochs, vals, color=colors[m], linewidth=2, label=f"m={m}")
            if ylog:
                ax.set_yscale("log")

    for ax, title in zip(axes, titles):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # reference lines
    axes[1].axhline(np.log(N_BINS), color="gray", linestyle="--",
                    linewidth=1, label="Uniform")
    axes[3].axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="No degradation")

    # final-epoch (theta, beta) scatter for largest width
    # use a 5th inset panel on top of prune ratio
    # final-epoch (theta, beta) scatter for largest width
    largest_m = max(snaps_by_width.keys())

    final = snaps_by_width[largest_m][-1]

    ax_s = fig.add_axes([0.55, 0.08, 0.18, 0.30])

    theta_f = np.array(final["theta"])
    beta_f  = np.array(final["beta"])

    _, lbl = offset_clustering(theta_f, beta_f)

    unique = sorted(set(lbl))
    cmap   = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))

    for i, l in enumerate(unique):
        mask = lbl == l
        c = "lightgray" if l == -1 else cmap[i % len(cmap)]
        ax_s.scatter(theta_f[mask], beta_f[mask], c=[c], s=8, alpha=0.6)

    ax_s.set_title(
        f"(θ,β) m={largest_m} ep={final['epoch']}",
        fontsize=8
    )

    ax_s.set_xticks([0, np.pi/2, np.pi])
    ax_s.set_xticklabels(["0", "π/2", "π"], fontsize=7)
    ax_s.tick_params(labelsize=7)


def plot_summary(all_results, out_dir):
    """
    3×3 grid: rows = diagnostics, cols = targets.
    Each cell shows the three width curves.
    Clean side-by-side comparison of all targets × all widths.
    """
    targets = list(all_results.keys())
    diag_keys   = ["angular_entropy", "n_clusters", "prune_ratio"]
    diag_labels = ["Angular Entropy", "Cluster Count", "Prune Ratio"]
    colors = {64: "#2563eb", 256: "#16a34a", 512: "#dc2626", 1024: "#000000"}

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("Parameter Collapse v2 — Full Summary (d=2)", fontsize=14, fontweight="bold")

    for col, tname in enumerate(targets):
        _, tlabel = TARGETS[tname]
        for row, (key, klabel) in enumerate(zip(diag_keys, diag_labels)):
            ax = axes[row, col]
            for m, snaps in all_results[tname].items():
                epochs = [s["epoch"] for s in snaps]
                vals   = [s[key] for s in snaps]
                ax.plot(epochs, vals, color=colors[m], linewidth=2, label=f"m={m}")
            if row == 0:
                ax.set_title(tlabel, fontsize=10)
                ax.axhline(np.log(N_BINS), color="gray", linestyle="--", linewidth=1)
            if row == 2:
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(klabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "collapse_v2_summary.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir     = "collapse_v2_plots"
    all_results = {}   # all_results[target_name][width] = snapshots

    for tname, (fn, label) in TARGETS.items():
        all_results[tname] = {}
        for m in WIDTHS:
            print(f"\n=== {label}  |  m={m} ===")
            all_results[tname][m] = train(fn, tname, m)
        plot_target_width(tname, label, all_results[tname], out_dir)

    print("\n=== Summary plot ===")
    plot_summary(all_results, out_dir)

    # save slim JSON (no point clouds)
    slim = {}
    for tname, by_width in all_results.items():
        slim[tname] = {}
        for m, snaps in by_width.items():
            slim[tname][str(m)] = [
                {k: v for k, v in s.items() if k not in ("theta","beta","alpha")}
                for s in snaps
            ]
    with open("collapse_v2_results.json", "w") as f:
        json.dump(slim, f, indent=2)
    print("Saved collapse_v2_results.json")


if __name__ == "__main__":
    main()