import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# =====================================================
# Config
# =====================================================

SEED = 0

p = 17
hidden_dim = 128

steps = 20000
record_every = 200

lr = 1e-3
weight_decay = 1e-2

train_fraction = 0.3

torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================
# Output directory
# =====================================================

output_dir = Path(
    f"results/p={p}_hidden={hidden_dim}_seed={SEED}"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print(f"\nSaving results to:\n{output_dir}\n")

# =====================================================
# Dataset
# =====================================================

data = []

for a in range(p):
    for b in range(p):
        x = [a, b]
        y = (a + b) % p
        data.append((x, y))

np.random.shuffle(data)

n_train = int(len(data) * train_fraction)

train_data = data[:n_train]
test_data = data[n_train:]

def make_tensor(dataset):

    x = torch.tensor(
        [d[0] for d in dataset],
        dtype=torch.long
    )

    y = torch.tensor(
        [d[1] for d in dataset],
        dtype=torch.long
    )

    return x, y

x_train, y_train = make_tensor(train_data)
x_test, y_test = make_tensor(test_data)

# =====================================================
# Model
# =====================================================

class ModularNet(nn.Module):

    def __init__(self, p, hidden_dim):
        super().__init__()

        self.embed = nn.Embedding(
            p,
            hidden_dim
        )

        self.fc1 = nn.Linear(
            2 * hidden_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            p
        )

    def forward(self, x):

        a = self.embed(x[:, 0])
        b = self.embed(x[:, 1])

        h = torch.cat(
            [a, b],
            dim=1
        )

        h = torch.relu(
            self.fc1(h)
        )

        out = self.fc2(h)

        return out

model = ModularNet(
    p,
    hidden_dim
)

# =====================================================
# Optimizer
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

loss_fn = nn.CrossEntropyLoss()

# =====================================================
# Helpers
# =====================================================

def accuracy(logits, y):

    pred = logits.argmax(dim=1)

    return (
        (pred == y)
        .float()
        .mean()
        .item()
    )

def bias_entropy(biases, bins=30):

    hist, _ = np.histogram(
        biases,
        bins=bins,
        density=True
    )

    hist += 1e-8

    hist = hist / hist.sum()

    return -np.sum(
        hist * np.log(hist)
    )

# =====================================================
# Storage
# =====================================================

records = []

bias_snapshots = []

# =====================================================
# Training
# =====================================================

for step in range(steps + 1):

    logits = model(x_train)

    loss = loss_fn(
        logits,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % record_every == 0:

        with torch.no_grad():

            train_logits = model(x_train)
            test_logits = model(x_test)

            train_loss = loss_fn(
                train_logits,
                y_train
            ).item()

            test_loss = loss_fn(
                test_logits,
                y_test
            ).item()

            train_acc = accuracy(
                train_logits,
                y_train
            )

            test_acc = accuracy(
                test_logits,
                y_test
            )

            biases = (
                model.fc1.bias
                .detach()
                .cpu()
                .numpy()
            )

            entropy = bias_entropy(
                biases
            )

            records.append({
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "bias_entropy": entropy
            })

            bias_snapshots.append(
                biases.copy()
            )

        print(
            f"step={step:5d} "
            f"train_acc={train_acc:.3f} "
            f"test_acc={test_acc:.3f}"
        )

# =====================================================
# Save metrics
# =====================================================

metrics_df = pd.DataFrame(records)

metrics_df.to_csv(
    output_dir / "metrics.csv",
    index=False
)

# =====================================================
# Save bias snapshots
# =====================================================

bias_df = pd.DataFrame(
    bias_snapshots
)

bias_df.to_csv(
    output_dir / "bias_snapshots.csv",
    index=False
)

# =====================================================
# Accuracy Plot
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    metrics_df["step"],
    metrics_df["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    metrics_df["step"],
    metrics_df["test_acc"],
    label="Test Accuracy"
)

plt.xlabel("Training Step")
plt.ylabel("Accuracy")

plt.title(
    "Grokking Check"
)

plt.legend()

plt.savefig(
    output_dir / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Loss Plot
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    metrics_df["step"],
    metrics_df["train_loss"],
    label="Train Loss"
)

plt.plot(
    metrics_df["step"],
    metrics_df["test_loss"],
    label="Test Loss"
)

plt.yscale("log")

plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.title(
    "Loss Curves"
)

plt.legend()

plt.savefig(
    output_dir / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Bias Entropy Plot
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    metrics_df["step"],
    metrics_df["bias_entropy"]
)

plt.xlabel("Training Step")

plt.ylabel("Entropy")

plt.title(
    "Bias Entropy"
)

plt.savefig(
    output_dir / "bias_entropy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Bias Dynamics Plot
# =====================================================

plt.figure(figsize=(10,6))

for i, biases in enumerate(bias_snapshots):

    y = np.ones_like(
        biases
    ) * i

    plt.scatter(
        biases,
        y,
        s=8,
        alpha=0.4
    )

plt.xlabel("Bias Value")
plt.ylabel("Checkpoint")

plt.title(
    "Bias Dynamics"
)

plt.savefig(
    output_dir / "bias_dynamics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFinished.\n")

print("Files generated:\n")

print(output_dir / "metrics.csv")
print(output_dir / "bias_snapshots.csv")

print(output_dir / "accuracy_curve.png")
print(output_dir / "loss_curve.png")
print(output_dir / "bias_entropy.png")
print(output_dir / "bias_dynamics.png")