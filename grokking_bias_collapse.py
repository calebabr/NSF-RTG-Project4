import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

SEED = 0

p = 17
hidden_dim = 128

steps = 20000
record_every = 200

lr = 1e-3
weight_decay = 1e-2
train_fraction = 0.3

train_acc_threshold = 0.99
test_acc_threshold = 0.95
collapse_drop_fraction = 0.20

torch.manual_seed(SEED)
np.random.seed(SEED)

output_dir = Path(f"results/p={p}_hidden={hidden_dim}_seed={SEED}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving results to:\n{output_dir}\n")

data = []

for a in range(p):
    for b in range(p):
        data.append(([a, b], (a + b) % p))

np.random.shuffle(data)

n_train = int(len(data) * train_fraction)
train_data = data[:n_train]
test_data = data[n_train:]

def make_tensor(dataset):
    x = torch.tensor([d[0] for d in dataset], dtype=torch.long)
    y = torch.tensor([d[1] for d in dataset], dtype=torch.long)
    return x, y

x_train, y_train = make_tensor(train_data)
x_test, y_test = make_tensor(test_data)

class ModularNet(nn.Module):
    def __init__(self, p, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(p, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, p)

    def forward(self, x):
        a = self.embed(x[:, 0])
        b = self.embed(x[:, 1])
        h = torch.cat([a, b], dim=1)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

model = ModularNet(p, hidden_dim)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

loss_fn = nn.CrossEntropyLoss()

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def bias_entropy(biases, bins=30):
    hist, _ = np.histogram(biases, bins=bins, density=True)
    hist += 1e-8
    hist = hist / hist.sum()
    return -np.sum(hist * np.log(hist))

records = []
bias_snapshots = []

for step in range(steps + 1):
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % record_every == 0:
        with torch.no_grad():
            train_logits = model(x_train)
            test_logits = model(x_test)

            train_loss = loss_fn(train_logits, y_train).item()
            test_loss = loss_fn(test_logits, y_test).item()

            train_acc = accuracy(train_logits, y_train)
            test_acc = accuracy(test_logits, y_test)

            biases = model.fc1.bias.detach().cpu().numpy()
            entropy = bias_entropy(biases)

            records.append({
                "step": step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "bias_entropy": entropy
            })

            bias_snapshots.append(biases.copy())

        print(
            f"step={step:5d} "
            f"train_acc={train_acc:.3f} "
            f"test_acc={test_acc:.3f} "
            f"bias_entropy={entropy:.3f}"
        )

metrics_df = pd.DataFrame(records)

metrics_df.to_csv(
    output_dir / "metrics.csv",
    index=False
)

bias_df = pd.DataFrame(bias_snapshots)

bias_df.to_csv(
    output_dir / "bias_snapshots.csv",
    index=False
)

# =====================================================
# Detect memorization time and grokking time
# =====================================================

def first_step_above(df, column, threshold):
    hit = df[df[column] >= threshold]
    if len(hit) == 0:
        return None
    return int(hit.iloc[0]["step"])

memorization_time = first_step_above(
    metrics_df,
    "train_acc",
    train_acc_threshold
)

grokking_time = first_step_above(
    metrics_df,
    "test_acc",
    test_acc_threshold
)

if memorization_time is not None and grokking_time is not None:
    grokking_delay = grokking_time - memorization_time
else:
    grokking_delay = None

grokking_df = pd.DataFrame([{
    "train_acc_threshold": train_acc_threshold,
    "test_acc_threshold": test_acc_threshold,
    "memorization_time": memorization_time,
    "grokking_time": grokking_time,
    "grokking_delay": grokking_delay
}])

grokking_df.to_csv(
    output_dir / "grokking_time.csv",
    index=False
)

# =====================================================
# Detect collapse time
# =====================================================

initial_entropy = metrics_df["bias_entropy"].iloc[0]
final_entropy = metrics_df["bias_entropy"].iloc[-1]

entropy_drop = initial_entropy - final_entropy

collapse_threshold = initial_entropy - collapse_drop_fraction * entropy_drop

collapse_candidates = metrics_df[
    metrics_df["bias_entropy"] <= collapse_threshold
]

if len(collapse_candidates) == 0:
    collapse_time = None
else:
    collapse_time = int(collapse_candidates.iloc[0]["step"])

if collapse_time is not None and grokking_time is not None:
    collapse_grokking_gap = grokking_time - collapse_time
else:
    collapse_grokking_gap = None

collapse_df = pd.DataFrame([{
    "initial_entropy": initial_entropy,
    "final_entropy": final_entropy,
    "entropy_drop": entropy_drop,
    "collapse_drop_fraction": collapse_drop_fraction,
    "collapse_threshold": collapse_threshold,
    "collapse_time": collapse_time,
    "grokking_time": grokking_time,
    "collapse_grokking_gap": collapse_grokking_gap
}])

collapse_df.to_csv(
    output_dir / "collapse_time.csv",
    index=False
)

# =====================================================
# Plot accuracy
# =====================================================

plt.figure(figsize=(8, 5))

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

if memorization_time is not None:
    plt.axvline(
        memorization_time,
        linestyle="--",
        label="Memorization Time"
    )

if grokking_time is not None:
    plt.axvline(
        grokking_time,
        linestyle="--",
        label="Grokking Time"
    )

plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.title("Grokking Check")
plt.legend()

plt.savefig(
    output_dir / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Plot loss
# =====================================================

plt.figure(figsize=(8, 5))

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
plt.title("Loss Curves")
plt.legend()

plt.savefig(
    output_dir / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Plot bias entropy
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    metrics_df["step"],
    metrics_df["bias_entropy"],
    label="Bias Entropy"
)

plt.axhline(
    collapse_threshold,
    linestyle="--",
    label="Collapse Threshold"
)

if collapse_time is not None:
    plt.axvline(
        collapse_time,
        linestyle="--",
        label="Collapse Time"
    )

if grokking_time is not None:
    plt.axvline(
        grokking_time,
        linestyle=":",
        label="Grokking Time"
    )

plt.xlabel("Training Step")
plt.ylabel("Entropy")
plt.title("Bias Collapse Proxy")
plt.legend()

plt.savefig(
    output_dir / "bias_entropy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Plot bias dynamics
# =====================================================

plt.figure(figsize=(10, 6))

for i, biases in enumerate(bias_snapshots):
    y = np.ones_like(biases) * i
    plt.scatter(
        biases,
        y,
        s=8,
        alpha=0.4
    )

plt.xlabel("Bias Value")
plt.ylabel("Checkpoint")
plt.title("Bias Dynamics")

plt.savefig(
    output_dir / "bias_dynamics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Save experiment config
# =====================================================

config_df = pd.DataFrame([{
    "seed": SEED,
    "p": p,
    "hidden_dim": hidden_dim,
    "steps": steps,
    "record_every": record_every,
    "lr": lr,
    "weight_decay": weight_decay,
    "train_fraction": train_fraction,
    "train_acc_threshold": train_acc_threshold,
    "test_acc_threshold": test_acc_threshold,
    "collapse_drop_fraction": collapse_drop_fraction
}])

config_df.to_csv(
    output_dir / "config.csv",
    index=False
)

print("\nFinished.\n")

print("Generated files:")
for file in output_dir.iterdir():
    print(file)

print("\nSummary:")
print(grokking_df)
print(collapse_df)