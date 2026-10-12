"""Tests for the head-scoring function on hand-built attention patterns with known answers.

Each fake pattern has shape [batch, heads, looking position, looked-at position], like the
real ones. Every row puts 100% of its attention on exactly one position, so the correct
score is known in advance (exactly 1 or exactly 0).
"""

import pytest
import torch

from induction_heads.scores import score

L = 5  # repeat length used in these tests
T = 2 * L  # positions per sequence: first copy at 0..L-1, second copy at L..2L-1


def prev_token_pattern() -> torch.Tensor:
    """A perfect previous-token head: every row i >= 1 attends 100% to position i - 1."""
    fake = torch.zeros(1, 1, T, T)
    fake[0, 0, 0, 0] = 1.0  # position 0 can only attend to itself
    rows = torch.arange(1, T)
    fake[0, 0, rows, rows - 1] = 1.0
    return fake


def induction_pattern() -> torch.Tensor:
    """A perfect induction head: rows in the second copy attend to position i - L + 1.

    Rows in the first copy have no earlier copy to use, so they attend to position 0
    (like the attention sink seen in the trained model).
    """
    fake = torch.zeros(1, 1, T, T)
    fake[0, 0, torch.arange(0, L), 0] = 1.0
    rows = torch.arange(L, T)
    fake[0, 0, rows, rows - (L - 1)] = 1.0
    return fake


def test_fake_patterns_are_valid():
    # Each row of a real attention pattern sums to 1; the fakes must too.
    for fake in (prev_token_pattern(), induction_pattern()):
        assert torch.allclose(fake.sum(dim=-1), torch.ones(1, 1, T))


def test_previous_token_head():
    fake = prev_token_pattern()
    prev = score(fake, torch.arange(1, T), 1)
    induction = score(fake, torch.arange(L, T), L - 1)
    dupe = score(fake, torch.arange(L, T), L)

    assert prev.shape == (1,)  # one score per head
    assert prev[0].item() == pytest.approx(1.0)
    assert induction[0].item() == pytest.approx(0.0)
    assert dupe[0].item() == pytest.approx(0.0)


def test_induction_head():
    fake = induction_pattern()
    induction = score(fake, torch.arange(L, T), L - 1)
    dupe = score(fake, torch.arange(L, T), L)

    assert induction[0].item() == pytest.approx(1.0)
    assert dupe[0].item() == pytest.approx(0.0)


def test_heads_are_scored_separately():
    # Two heads in one pattern: head 0 previous-token, head 1 induction.
    # Checks that scores come back in head order and don't mix heads together.
    fake = torch.cat([prev_token_pattern(), induction_pattern()], dim=1)  # [1, 2, T, T]
    induction = score(fake, torch.arange(L, T), L - 1)

    assert induction.shape == (2,)
    assert induction[0].item() == pytest.approx(0.0)
    assert induction[1].item() == pytest.approx(1.0)


def test_batch_is_averaged():
    # A batch of 3 identical sequences must score the same as 1 sequence.
    fake = prev_token_pattern().repeat(3, 1, 1, 1)  # [3, 1, T, T]
    prev = score(fake, torch.arange(1, T), 1)

    assert prev[0].item() == pytest.approx(1.0)
