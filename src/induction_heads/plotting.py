"""Figures: loss curves and attention-pattern grids."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files without opening windows

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

# Colors
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
FIRST_HALF_COLOR = "#2a78d6"  # blue
SECOND_HALF_COLOR = "#eb6834"  # orange
ATTENTION_CMAP = LinearSegmentedColormap.from_list(  # one hue, light (0) -> dark (1)
    "attention", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def get_attention_patterns(model: torch.nn.Module) -> list[torch.Tensor]:
    """Attention patterns from the model's last forward pass, one tensor per layer.

    Each tensor has shape [batch, n_heads, n_ctx, n_ctx]:
    row = destination position, column = source position, each row sums to 1.

    Assumes each attention layer is in `model.layers` and stores its softmax output
    as `.pattern`. If your model names these differently, change only this function.
    """
    return [layer.pattern.detach().cpu() for layer in model.layers]


def plot_loss_curves(
    steps: list[int],
    first_half: list[float],
    second_half: list[float],
    vocab_size: int,
    path: str | Path,
) -> None:
    """Loss on the random first half vs the repeated second half over training.

    The dashed line is ln(vocab_size): the loss of guessing uniformly at random.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    chance = math.log(vocab_size)
    ax.axhline(chance, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(  # below the line, mid-plot, clear of both curves
        steps[len(steps) // 2], chance - 0.08, f"chance: ln({vocab_size}) = {chance:.2f}",
        va="top", ha="center", fontsize=8, color=INK_SECONDARY,
    )

    for values, color, label in (
        (first_half, FIRST_HALF_COLOR, "First half (random)"),
        (second_half, SECOND_HALF_COLOR, "Second half (repeated)"),
    ):
        ax.plot(steps, values, color=color, linewidth=2, label=label, solid_capstyle="round")
        ax.annotate(  # direct label at the end of each line
            label, (steps[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8, color=INK,
        )

    ax.set_ylim(bottom=0, top=chance * 1.15)
    ax.set_xlabel("Training step", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Loss (cross-entropy)", color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Next-token loss by half of the sequence", color=INK, fontsize=11, loc="left")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_attention_patterns(
    patterns: list[torch.Tensor],
    path: str | Path,
    example: int = 0,
    half_len: int | None = None,
) -> None:
    """Grid of attention patterns: one row per layer, one column per head.

    patterns: output of get_attention_patterns(model), each [batch, n_heads, n_ctx, n_ctx].
    example:  which sequence in the batch to show.
    half_len: if given, thin lines mark where the repeated half starts.

    What to look for:
      - previous-token head: a bright line just below the diagonal (each position
        attends to the position before it).
      - induction head: in the second half, a bright line half_len - 1 positions
        left of the diagonal (each position attends to the token *after* its
        earlier copy).
    """
    n_layers, n_heads = len(patterns), patterns[0].shape[1]
    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(2.6 * n_heads, 2.6 * n_layers), dpi=200, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)

    for layer, layer_patterns in enumerate(patterns):
        for head in range(n_heads):
            ax = axes[layer][head]
            image = ax.imshow(
                layer_patterns[example, head].float().numpy(),
                cmap=ATTENTION_CMAP, vmin=0, vmax=1, interpolation="nearest",
            )
            if half_len is not None:
                ax.axhline(half_len - 0.5, color=INK_SECONDARY, linewidth=0.6)
                ax.axvline(half_len - 0.5, color=INK_SECONDARY, linewidth=0.6)
            ax.set_title(f"L{layer}H{head}", fontsize=9, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            if head == 0:
                ax.set_ylabel("destination", fontsize=8, color=INK_SECONDARY)
            if layer == n_layers - 1:
                ax.set_xlabel("source", fontsize=8, color=INK_SECONDARY)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
    colorbar.set_label("attention weight", fontsize=8, color=INK_SECONDARY)
    colorbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
