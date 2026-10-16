"""Figures for the README: what the heads do (Fig 1) and ablations (Fig 2).

Plumbing only (AI-written). Uses the same model, batch and seed as 05_ablations.py.
Saves PNG (README) + PDF (portfolio) to figures/.
"""

import math

import torch
import torch.nn.functional as F

from induction_heads.config import Config, get_device
from induction_heads.data import make_repeated_batch
from induction_heads.model import AttnOnlyAutobot
from induction_heads.plotting import (
    get_attention_patterns,
    plot_ablations,
    plot_head_scores,
    plot_selected_heads,
)
from induction_heads.scores import score

L = 20  # repeat length
HEAD_LABELS = [f"L{layer}H{head}" for layer in range(2) for head in range(4)]

# --- Model + batch (same as 05_ablations.py) -------------------------------------

cfg = Config(n_steps=20000)
device = get_device()
model = AttnOnlyAutobot(cfg).to(device)
model.load_state_dict(torch.load("runs/varlen_synthetic.pt", map_location=device))
model.eval()

gen = torch.Generator().manual_seed(0)
tokens = make_repeated_batch(cfg, batch_size=256, half_len=L, generator=gen, device=device)

with torch.no_grad():
    logits = model(tokens)
patterns = get_attention_patterns(model)  # one [batch, heads, n_ctx, n_ctx] per layer, on CPU
CHANCE = math.log(logits.size(-1))

# Sanity check: should match 05_ablations.py's baseline (repeated 0.061, first 6.38).
V = logits.size(-1)
rep = F.cross_entropy(logits[:, L:2 * L - 1].reshape(-1, V), tokens[:, L + 1:2 * L].reshape(-1))
first = F.cross_entropy(logits[:, :L - 1].reshape(-1, V), tokens[:, 1:L].reshape(-1))
print(f"baseline: repeated {rep.item():.3f}   first {first.item():.2f}   chance {CHANCE:.2f}")

# --- Figure 1a: one head per role ------------------------------------------------

plot_selected_heads(
    patterns,
    heads=[(0, 2, "previous-token"), (0, 1, "duplicate-token"), (1, 2, "induction")],
    path="figures/fig1a_attention_by_role.png",
    example=0,
    half_len=L,
)

# --- Figure 1b: head scores ------------------------------------------------------
# Same definitions as 04_head_scores.py, but on this 256-sequence batch
# (04 used 64 sequences), so values can differ from the README table in the 2nd decimal.

prev_rows = torch.arange(1, 2 * L)   # whole sequence, attend to i - 1
second_rows = torch.arange(L, 2 * L)  # second copy only

scores = {"previous-token": [], "duplicate-token": [], "induction": []}
for layer_pattern in patterns:
    scores["previous-token"] += score(layer_pattern, prev_rows, 1).tolist()
    scores["duplicate-token"] += score(layer_pattern, second_rows, L).tolist()
    scores["induction"] += score(layer_pattern, second_rows, L - 1).tolist()

for i, label in enumerate(HEAD_LABELS):
    print(label, "  ".join(f"{name} {values[i]:.3f}" for name, values in scores.items()))

plot_head_scores(
    scores,
    HEAD_LABELS,
    path="figures/fig1b_head_scores.png",
    title=f"Head scores (repeat length {L})",
    row_breaks=(4,),
)

# --- Figure 2: ablations ---------------------------------------------------------
# Numbers from 05_ablations.py (seed 0, 256 sequences, length 20): (repeated, first).

BASELINE = (0.061, 6.38)
RESULTS = [
    ("Previous-token\n(L0H2, L0H3)", {"zero": (1.393, 6.28), "mean": (1.196, 6.28)}),
    ("Duplicate-token\n(L0H0, L0H1)", {"zero": (8.245, 10.15), "mean": (0.294, 6.35)}),
    ("All of\nlayer 0", {"zero": (6.867, 6.95), "mean": (2.319, 6.28)}),
    ("All of\nlayer 1", {"zero": (8.560, 9.85), "mean": (8.448, 9.75)}),
]

plot_ablations(
    RESULTS,
    BASELINE,
    CHANCE,
    path="figures/fig2_ablations.png",
    title=f"Silencing heads: zero- vs mean-ablation (repeat length {L})",
)
print("Saved figures/fig1a_attention_by_role, fig1b_head_scores, fig2_ablations (.png + .pdf)")
