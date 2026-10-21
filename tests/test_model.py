"""Sanity checks for the forward pass. Run from the repo root: uv run pytest

These assume the model lives in src/induction_heads/model.py as:
    class AttnOnlyAutobot(nn.Module):
        def __init__(self, cfg: Config): ...
        def forward(self, tokens):  # tokens [batch, n_ctx] -> logits [batch, n_ctx, vocab_size]
with its attention layers in `self.layers`, each storing its softmax output as `.pattern`.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from induction_heads.config import Config
from induction_heads.data import make_repeated_batch
from induction_heads.model import AttnOnlyAutobot
from induction_heads.plotting import get_attention_patterns

CFG = Config()
BATCH = 8


@pytest.fixture
def model() -> AttnOnlyAutobot:
    torch.manual_seed(0)
    return AttnOnlyAutobot(CFG).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
    return make_repeated_batch(CFG, batch_size=BATCH, generator=torch.Generator().manual_seed(0))


def test_output_shape(model, tokens):
    logits = model(tokens)
    assert logits.shape == (BATCH, CFG.n_ctx, CFG.vocab_size)


def test_causality(model, tokens):
    """Changing tokens from position t onward must not change the logits before t."""
    t = CFG.n_ctx // 2
    changed = tokens.clone()
    changed[:, t:] = (changed[:, t:] + 1) % CFG.vocab_size

    with torch.no_grad():
        original_logits = model(tokens)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :t], changed_logits[:, :t],
        msg="Logits before position t changed when only later tokens changed: "
        "the causal mask is missing or wrong (the model can see the future).",
    )
    assert not torch.allclose(original_logits[:, t:], changed_logits[:, t:]), (
        "Logits from position t onward didn't change when their tokens did: "
        "the model may be ignoring its input."
    )


def test_initial_loss(model, tokens):
    """An untrained model should be close to uniform guessing: loss ~ ln(vocab_size)."""
    with torch.no_grad():
        logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, CFG.vocab_size), tokens[:, 1:].reshape(-1)
    ).item()
    expected = math.log(CFG.vocab_size)  # ln(512) ~ 6.24
    assert abs(loss - expected) < 0.5, (
        f"Initial loss {loss:.2f}, expected about ln({CFG.vocab_size}) = {expected:.2f}. "
        "Much higher usually means the initial weights are too large "
        "(logits far from 0 make confident wrong guesses)."
    )


def test_attention_patterns(model, tokens):
    """One pattern per layer; each row sums to 1; nothing attends to later positions."""
    with torch.no_grad():
        model(tokens)
    patterns = get_attention_patterns(model)
    assert len(patterns) == CFG.n_layers

    future = torch.triu(torch.ones(CFG.n_ctx, CFG.n_ctx, dtype=torch.bool), diagonal=1)
    for layer, pattern in enumerate(patterns):
        assert pattern.shape == (BATCH, CFG.n_heads, CFG.n_ctx, CFG.n_ctx), f"layer {layer}"
        torch.testing.assert_close(
            pattern.sum(dim=-1), torch.ones(BATCH, CFG.n_heads, CFG.n_ctx),
            msg=f"Layer {layer}: attention rows don't sum to 1 (softmax over the wrong dimension?)",
        )
        assert pattern[..., future].abs().max() < 1e-6, (
            f"Layer {layer}: positions attend to later positions (causal mask not applied)"
        )
