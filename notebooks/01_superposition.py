"""
Toy Models of Superposition: warm-up replication.

Question: when a network has more features than hidden dimensions, which
features does it keep, and how does it arrange them?

Model (from the paper):
    h  = W x                 # compress: N_FEATURES -> N_HIDDEN
    x' = ReLU(W^T h + b)     # reconstruct: N_HIDDEN -> N_FEATURES
Loss: importance-weighted squared reconstruction error.

We train one model per sparsity level and plot where each feature ends up
in the hidden space.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
N_FEATURES = 5           # number of input features
N_HIDDEN = 2             # size of the bottleneck (hidden dimensions); 2 for plotting
IMPORTANCE_DECAY = 0.9   # importance of feature i = IMPORTANCE_DECAY ** i (feature 0 matters most)
SPARSITY_LEVELS = [0.0, 0.7, 0.9, 0.97, 0.99]  # probability that a feature is zero in a sample
BATCH_SIZE = 1024        # samples per training step
TRAINING_STEPS = 10_000  # optimizer steps per model
LEARNING_RATE = 1e-3     # Adam learning rate
LOG_EVERY = 1_000        # record the loss every this many steps
SEED = 0                 # random seed, so runs are repeatable
DEVICE = "cpu"           # no need for gpu
FIGURE_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Data (plumbing)
# ---------------------------------------------------------------------------
def make_importance(n_features: int, decay: float) -> torch.Tensor:
    """Importance weight per feature: [1, decay, decay^2, ...]. Shape (n_features,)."""
    return decay ** torch.arange(n_features, dtype=torch.float32)


def generate_batch(batch_size: int, n_features: int, feature_probability: float,
                   device: str = DEVICE) -> torch.Tensor:
    """
    Sparse synthetic data. Shape (batch_size, n_features).
    Each feature is independently "on" with probability feature_probability
    (= 1 - sparsity). When on, its value is uniform in [0, 1); otherwise 0.
    """
    values = torch.rand(batch_size, n_features, device=device)
    is_on = (torch.rand(batch_size, n_features, device=device) < feature_probability).float()
    return values * is_on


# ---------------------------------------------------------------------------
# Model 
# ---------------------------------------------------------------------------
class ToyModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        
        # init params
        self.W = nn.Parameter(torch.empty(n_hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))

        nn.init.xavier_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # forward pass
        h = x @ self.W.T
        x_hat = torch.relu(h @ self.W + self.b)

        return x_hat

# ---------------------------------------------------------------------------
# Training 
# ---------------------------------------------------------------------------
def train(model: ToyModel, feature_probability: float,
          importance: torch.Tensor) -> list[float]:
    """
    Train `model` to reconstruct sparse inputs. Return the list of loss
    values recorded every LOG_EVERY steps (main() prints the last one).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    loss_per_step = []

    for step in range (1, TRAINING_STEPS + 1):
        batch = generate_batch(BATCH_SIZE, N_FEATURES, feature_probability)

        optimizer.zero_grad()

        x_hat = model(batch)

        sq_error = (batch - x_hat)**2

        weighted_sum = (sq_error * importance).sum(dim=1)
        loss = weighted_sum.mean()

        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            loss_per_step.append(loss.item())

    return loss_per_step

# ---------------------------------------------------------------------------
# Analysis and plots (plumbing)
# ---------------------------------------------------------------------------
def plot_feature_directions(results, importance: torch.Tensor) -> None:
    """One panel per sparsity level: each feature's column of W as an arrow."""
    assert N_HIDDEN == 2, "This plot needs exactly 2 hidden dimensions."
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.6))
    cmap = plt.get_cmap("viridis")

    for panel, (ax, (sparsity, model, _)) in enumerate(zip(axes, results)):
        W = model.W.detach().cpu()
        limit = max(1.2, W.norm(dim=0).max().item() * 1.15)
        for i in range(N_FEATURES):
            color = cmap(i / max(N_FEATURES - 1, 1))
            label = f"feature {i} (importance {importance[i]:.2f})" if panel == 0 else None
            ax.plot([0, W[0, i]], [0, W[1, i]], color=color, linewidth=2, label=label)
            ax.scatter(W[0, i], W[1, i], color=color, s=20, zorder=3)
        ax.axhline(0, color="lightgray", linewidth=0.5)
        ax.axvline(0, color="lightgray", linewidth=0.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(f"sparsity = {sparsity}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(loc="lower center", ncol=min(N_FEATURES, 3), fontsize=8, frameon=False)
    fig.suptitle(f"Where {N_FEATURES} features land in {N_HIDDEN} hidden dimensions")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, "superposition_directions.png")


def plot_gram_matrices(results) -> None:
    """
    W^T W for each model, shape (N_FEATURES, N_FEATURES).
    Diagonal entry i  = squared length of feature i (about 1 = represented, 0 = dropped).
    Off-diagonal i,j  = overlap between features i and j (0 = no interference).
    """
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    image = None
    for ax, (sparsity, model, _) in zip(axes, results):
        W = model.W.detach().cpu()
        gram = W.T @ W
        image = ax.imshow(gram, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title(f"sparsity = {sparsity}")
        ax.set_xticks(range(N_FEATURES))
        ax.set_yticks(range(N_FEATURES))
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.suptitle("W^T W: diagonal = feature length, off-diagonal = interference")
    _save(fig, "superposition_gram.png")


def print_feature_norms(results) -> None:
    """Table of each feature's length (norm of its column of W) and its bias."""
    print("\nFeature lengths ||W_i|| (about 1 = represented, about 0 = dropped) and biases b_i")
    header = "sparsity | " + " ".join(f"   f{i}  " for i in range(N_FEATURES))
    print(header)
    print("-" * len(header))
    for sparsity, model, _ in results:
        norms = model.W.detach().cpu().norm(dim=0)
        biases = model.b.detach().cpu()
        print(f"{sparsity:8.2f} | " + " ".join(f"{n:7.3f}" for n in norms) + "   (norm)")
        print(f"{'':8} | " + " ".join(f"{b:7.3f}" for b in biases) + "   (bias)")


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    importance = make_importance(N_FEATURES, IMPORTANCE_DECAY).to(DEVICE)

    results = []
    for sparsity in SPARSITY_LEVELS:
        feature_probability = 1 - sparsity
        model = ToyModel(N_FEATURES, N_HIDDEN).to(DEVICE)
        losses = train(model, feature_probability, importance)
        print(f"sparsity = {sparsity:.2f}   final loss = {losses[-1]:.6f}")
        results.append((sparsity, model, losses))

    print_feature_norms(results)
    plot_feature_directions(results, importance.cpu())
    plot_gram_matrices(results)
    plt.show()


if __name__ == "__main__":
    main()
