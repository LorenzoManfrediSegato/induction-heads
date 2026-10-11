"""Synthetic data (repeated random sequences) and per-half loss for logging."""

import torch
import torch.nn.functional as F

from induction_heads.config import Config


def sample_half_len(cfg: Config, generator: torch.Generator | None = None) -> int:
    """A random first-half length between cfg.min_half_len and cfg.half_len (inclusive).

    Training with a different length every batch means the copy doesn't always start
    the same number of positions later, so the model can't solve the task by position
    alone ("look 31 back") and has to match tokens by content (induction).
    """
    return int(torch.randint(cfg.min_half_len, cfg.half_len + 1, (1,), generator=generator).item())


def make_repeated_batch(
    cfg: Config,
    batch_size: int | None = None,
    half_len: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A batch of sequences: half_len random tokens, then the same tokens again.

    Returns integer token IDs, shape [batch_size, 2 * half_len].
    Example with half_len 4: [17, 3, 99, 42, 17, 3, 99, 42]

    half_len defaults to cfg.half_len (32). Pass sample_half_len(cfg) for a random length.
    Tokens are generated on the CPU (so a seeded `generator` gives the same data on
    any machine) and then moved to `device`.
    """
    b = cfg.batch_size if batch_size is None else batch_size
    n = cfg.half_len if half_len is None else half_len
    first_half = torch.randint(0, cfg.vocab_size, (b, n), generator=generator)
    return torch.cat([first_half, first_half], dim=1).to(device)


@torch.no_grad()
def loss_by_half(logits: torch.Tensor, tokens: torch.Tensor, half_len: int) -> dict[str, float]:
    """Next-token loss split by which part of the sequence is being predicted.

    For logging only; the training loss itself is computed in the training loop.
    Pass the half_len the batch was made with.

    The logits at position j predict the token at position j + 1. With half_len 32:
      - "first_half":  predicting tokens 1..31  (random: loss should stay near ln(vocab_size))
      - "boundary":    predicting token 32      (first token of the copy: unpredictable)
      - "second_half": predicting tokens 33..63 (copies: loss should fall towards 0)
    Token 0 is never predicted (nothing comes before it).

    logits: [batch, seq, vocab_size]; tokens: [batch, seq]
    """
    vocab_size = logits.shape[-1]
    per_position = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size).float(),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(tokens.shape[0], -1)  # [batch, seq - 1]; column j = loss predicting token j + 1

    return {
        "first_half": per_position[:, : half_len - 1].mean().item(),
        "boundary": per_position[:, half_len - 1].mean().item(),
        "second_half": per_position[:, half_len:].mean().item(),
    }
