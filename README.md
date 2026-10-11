# induction-heads

Training a 2-layer attention-only transformer from scratch and finding — then causally verifying — the induction-head circuit described in Anthropic's [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) and [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

> **Status: in progress.** The superposition warm-up is done; the induction-heads model is being built. This README is updated as results come in.

## Why this project

The question I keep coming back to is how networks learn and represent things — how simple units give rise to understanding. I first ran into it with robot swarms ([PiELo](TODO-link), my undergraduate thesis), where small local rules build into global behaviour. Induction heads are the cleanest example of the same thing inside a transformer: two attention heads, each doing something simple, combine into a general "copy what came next last time" algorithm, and that algorithm is a large part of how models learn in context.

The goal is to build the model myself, watch the circuit form, and prove it does what I think it does — not just by looking at attention patterns, but by knocking it out and measuring what breaks.

## What an induction head does

Given a sequence like `... A B ... A`, an induction head predicts `B`. It needs two heads working together:

1. **Previous-token head (layer 0):** at each position, copies in information about the token just before it. After this, the position holding `B` also "knows" it came after `A`.
2. **Induction head (layer 1):** at the current `A`, looks back for a position whose previous token was `A` — that's the earlier `B` — and copies `B` into the prediction.

## Plan

| Step | What | Status |
|---|---|---|
| 0 | Warm-up: reproduce the *Toy Models of Superposition* result | Done |
| 1 | Build the model (embeddings, attention, residual stream; no MLPs) + training loop | Done |
| 2 | Train on synthetic repeated sequences; confirm the loss drop on the repeated half | Done |
| 3 | Plot attention patterns; identify the previous-token head (L0) and induction head (L1) | Done |
| 4 | Causal test: ablate the induction head and show loss on the repeated half goes back up | Not started |
| 5 | Repeat on natural text | Not started |

## Results

### Warm-up: superposition

Reproduced the core result of *Toy Models of Superposition* (5 features squeezed into 2 dimensions):

- **Dense features:** the model keeps 2 features at right angles and drops the other 3.
- **Sparser features:** features pair up as antipodal pairs (pointing in opposite directions).
- **Sparsest features:** all 5 are stored, arranged as a pentagon.

Code: [`notebooks/01_superposition.py`](notebooks/01_superposition.py). Figures: [`figures/`](figures/). <!-- TODO: embed the key figure -->

### First attempt: the model found a positional shortcut

My first training run used sequences of 32 random tokens repeated once (64 tokens total). It looked like a success: the loss on the repeated half fell from 6.24 (random guessing over 512 tokens) to 0.004 in a sudden drop between steps ~400 and ~650, the phase change the induction-heads paper describes.

![Loss curves, fixed repeat length](figures/fixed_len_loss.png)

The attention patterns didn't match an induction circuit, though:

- **Layer-0 heads attended to the earlier copy.** Real induction needs two layers: layer 0 has no way to know which earlier position comes *after* a matching token, because that information only exists once a previous-token head has moved it.
- **No previous-token head.** No head attended to position i − 1 across the whole sequence, which is the first half of the circuit.

![Attention patterns, fixed repeat length](figures/fixed_len_attention.png)

Because the copy always started exactly 32 positions later, the model could solve the task by position alone: "at position i, copy whatever is at position i − 31". No token matching required, and one layer is enough.

**Test:** I gave the trained model sequences that repeat after 20 tokens instead of 32. A real induction circuit matches tokens by content, so it should work at any distance. The repeated-half loss was **11.4**: worse than random guessing (6.24), because the model confidently copied the token 31 positions back, which was now the wrong one. Shortcut confirmed.

**Fix:** the repeat length now changes every batch (8 to 32 tokens), so position no longer tells the model where the earlier copy is. Code for this run: [`notebooks/02_induction_synthetic.py`](notebooks/02_induction_synthetic.py), at commit `TODO-commit-hash`.

### Induction heads

_Coming soon: results with variable repeat length, attention patterns, ablation._

## Model

A 2-layer, attention-only transformer (4 heads per layer, `d_model` 64, ~102k parameters), trained on random token sequences that repeat. Full configuration, and the reasoning behind each choice, in [`MODEL_CONFIG.md`](MODEL_CONFIG.md).

## Repository layout

```
induction-heads/
├── src/induction_heads/
│   ├── config.py     # model + training settings (see MODEL_CONFIG.md)
│   ├── data.py       # repeated random sequences; loss split by half for logging
│   ├── model.py      # the 2-layer attention-only transformer
│   └── plotting.py   # loss curves, attention-pattern grids
├── tests/            # sanity checks: output shape, causality, initial loss
├── notebooks/        # runnable experiments (01_superposition.py, ...)
├── notes/            # reading notes, numbered (01-zoom-in.md, ...)
├── figures/          # saved plots
├── MODEL_CONFIG.md
└── README.md
```

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Runs on Apple Silicon (PyTorch MPS) or CPU.

```bash
git clone https://github.com/TODO-username/induction-heads.git
cd induction-heads
uv sync
uv run python notebooks/01_superposition.py   # superposition warm-up
uv run pytest                                  # model sanity checks
```
<!-- TODO: add the training command -->

## How I used AI

This is a learning project, so I set rules for where AI was and wasn't allowed:

- **Written by hand, without AI:** the model (embeddings, attention, the forward pass), the training loop, and the causal experiments (ablations). These are typed out by hand, and I struggle with any issues I may have for ~30 minutes before looking something up.
- **AI-assisted:** "plumbing" that isn't the point of the exercise — config, synthetic data generation, plotting, logging, figure styling. I've read and can explain every line of it.
- **Learning and notes:** I worked through the papers with Claude, asking questions as I read. The files in [`notes/`](notes/) were compiled from those conversations, parts written by me, parts by Claude; open questions I haven't resolved yet are marked as such.
- **This README** was drafted with Claude and edited by me.

## References

- Olah et al., [*Zoom In: An Introduction to Circuits*](https://distill.pub/2020/circuits/zoom-in/) (2020)
- Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021)
- Olsson et al., [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022)
- Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (2022)

## License

<!-- TODO: pick one (MIT is the common choice for a project like this) -->
