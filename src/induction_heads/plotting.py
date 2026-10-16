"""Figures: loss curves, attention-pattern grids, head scores and ablations."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files without opening windows

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

# Colors
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
FIRST_HALF_COLOR = "#2a78d6"  # blue
SECOND_HALF_COLOR = "#eb6834"  # orange
MEAN_COLOR = "#1c5cab"  # mean-ablation: the readable numbers
ZERO_COLOR = "#c9c7c0"  # zero-ablation: shown for contrast
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


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Save as the given file (e.g. .png for the README) plus a .pdf copy for the portfolio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


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


def plot_selected_heads(
    patterns: list[torch.Tensor],
    heads: list[tuple[int, int, str]],
    path: str | Path,
    example: int = 0,
    half_len: int | None = None,
) -> None:
    """Attention patterns of a few chosen heads, side by side, labelled by role.

    patterns: output of get_attention_patterns(model).
    heads:    (layer, head, role) per panel, e.g. [(0, 2, "previous-token"), ...].
    half_len: repeat length L; thin lines mark where the repeated half starts.
    """
    fig, axes = plt.subplots(
        1, len(heads), figsize=(3.2 * len(heads), 3.4), dpi=200, squeeze=False,
        layout="constrained",  # keeps the title close to the panels
    )
    fig.patch.set_facecolor(SURFACE)

    for ax, (layer, head, role) in zip(axes[0], heads):
        image = ax.imshow(
            patterns[layer][example, head].float().numpy(),
            cmap=ATTENTION_CMAP, vmin=0, vmax=1, interpolation="nearest",
        )
        if half_len is not None:
            ax.axhline(half_len - 0.5, color=INK_SECONDARY, linewidth=0.6)
            ax.axvline(half_len - 0.5, color=INK_SECONDARY, linewidth=0.6)
        ax.set_title(f"L{layer}H{head}\n{role}", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("source", fontsize=8, color=INK_SECONDARY)
    axes[0][0].set_ylabel("destination", fontsize=8, color=INK_SECONDARY)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    colorbar.set_label("attention weight", fontsize=8, color=INK_SECONDARY)
    colorbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)

    title = "Attention on a repeated random sequence"
    if half_len is not None:
        title += f" (repeat length {half_len})"
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    _save(fig, path)


def plot_head_scores(
    scores: dict[str, list[float]],
    head_labels: list[str],
    path: str | Path,
    title: str,
    row_breaks: tuple[int, ...] = (),
) -> None:
    """Heatmap of head scores, one row per head, one column per score.

    scores:      {score name: one value per head, in the order of head_labels}.
    row_breaks:  row indices to draw a gap above (e.g. (4,) to split layer 0 from layer 1).
    """
    names = list(scores)
    grid = np.array([scores[name] for name in names]).T  # [heads, scores]

    fig, ax = plt.subplots(
        figsize=(1.4 * len(names) + 1.6, 0.42 * len(head_labels) + 1.4), dpi=200
    )
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(grid, cmap=ATTENTION_CMAP, vmin=0, vmax=1, aspect="auto")

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color=SURFACE if value > 0.55 else INK,
            )
    for row in row_breaks:
        ax.axhline(row - 0.5, color=SURFACE, linewidth=3)

    ax.set_xticks(range(len(names)), names, fontsize=8, color=INK)
    ax.set_yticks(range(len(head_labels)), head_labels, fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=26)
    _save(fig, path)


def plot_ablations(
    results: list[tuple[str, dict[str, tuple[float, float]]]],
    baseline: tuple[float, float],
    chance: float,
    path: str | Path,
    title: str,
) -> None:
    """Zero- vs mean-ablation: repeated-half loss and first-half loss, side by side.

    results:  [(label, {"zero": (repeated, first), "mean": (repeated, first)}), ...]
    baseline: (repeated, first) loss of the unablated model on the same batch.
    chance:   ln(vocab_size).
    """
    labels = [label for label, _ in results]
    x = np.arange(len(results))
    width = 0.38

    fig, (ax_rep, ax_first) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=200, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)

    panels = (
        (ax_rep, 0, "Repeated half (low = induction still works)"),
        (ax_first, 1, "Random first half (sanity check: should stay near chance)"),
    )
    for ax, idx, subtitle in panels:
        _style(ax)
        for offset, mode, color in ((-width / 2, "zero", ZERO_COLOR), (width / 2, "mean", MEAN_COLOR)):
            values = [modes[mode][idx] for _, modes in results]
            bars = ax.bar(x + offset, values, width, color=color, label=f"{mode}-ablation")
            ax.bar_label(
                bars, labels=[f"{v:.2f}" for v in values], fontsize=7,
                color=INK_SECONDARY, padding=2,
            )
        # Reference lines are named in the shared legend (labelled on the left panel only),
        # so no text sits on top of the bars.
        first_panel = idx == 0
        ax.axhline(
            chance, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
            label=f"chance (ln vocab = {chance:.2f})" if first_panel else "_nolegend_",
        )
        ax.axhline(
            baseline[idx], color=INK, linewidth=1.2, linestyle=(0, (1, 2)),
            label="no ablation" if first_panel else "_nolegend_",
        )
        ax.set_xticks(x, labels, fontsize=8, color=INK)
        ax.set_ylabel("Loss (cross-entropy)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(subtitle, loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    all_rep = [modes[m][0] for _, modes in results for m in ("zero", "mean")] + [baseline[0]]
    all_first = [modes[m][1] for _, modes in results for m in ("zero", "mean")] + [baseline[1]]
    ax_rep.set_yscale("log")
    ax_rep.set_ylim(min(all_rep) / 2, max(all_rep) * 3)
    ax_first.set_ylim(0, max(all_first) * 1.15)

    fig.legend(
        *ax_rep.get_legend_handles_labels(), loc="outside upper right", ncol=4,
        frameon=False, fontsize=8,
    )
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    _save(fig, path)
