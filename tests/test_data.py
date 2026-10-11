"""Checks for the data and logging helpers. Run from the repo root: uv run pytest"""

import math

import torch

from induction_heads.config import Config
from induction_heads.data import loss_by_half, make_repeated_batch, sample_half_len

CFG = Config()


def test_batch_is_repeated():
    batch = make_repeated_batch(CFG, generator=torch.Generator().manual_seed(0))
    assert batch.shape == (CFG.batch_size, CFG.n_ctx)
    assert batch.dtype == torch.long
    assert torch.equal(batch[:, : CFG.half_len], batch[:, CFG.half_len :])
    assert batch.min() >= 0 and batch.max() < CFG.vocab_size


def test_custom_half_len():
    batch = make_repeated_batch(CFG, batch_size=4, half_len=10)
    assert batch.shape == (4, 20)
    assert torch.equal(batch[:, :10], batch[:, 10:])


def test_sample_half_len_in_range():
    g = torch.Generator().manual_seed(0)
    lengths = {sample_half_len(CFG, generator=g) for _ in range(500)}
    assert min(lengths) == CFG.min_half_len
    assert max(lengths) == CFG.half_len


def test_seeded_batches_match():
    a = make_repeated_batch(CFG, generator=torch.Generator().manual_seed(1))
    b = make_repeated_batch(CFG, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_loss_by_half_uniform():
    """All-zero logits = uniform guessing, so every part should be ln(vocab_size)."""
    tokens = make_repeated_batch(CFG, batch_size=4)
    logits = torch.zeros(4, CFG.n_ctx, CFG.vocab_size)
    losses = loss_by_half(logits, tokens, CFG.half_len)
    for value in losses.values():
        assert math.isclose(value, math.log(CFG.vocab_size), rel_tol=1e-5)
