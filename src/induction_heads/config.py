"""Model and training configuration. Values and reasoning: see MODEL_CONFIG.md."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    # Data
    vocab_size: int = 512  # number of distinct tokens
    half_len: int = 32  # longest random first half; the second half is a copy
    min_half_len: int = 8  # shortest random first half (each batch picks a length in between)

    # Model
    n_layers: int = 2  # attention layers
    d_model: int = 64  # residual stream width
    n_heads: int = 4  # attention heads per layer

    # Training
    batch_size: int = 64  # sequences per training step
    lr: float = 1e-3  # learning rate
    weight_decay: float = 0.01  # AdamW weight decay
    n_steps: int = 20_000  # training steps
    log_every: int = 50  # record losses every this many steps
    seed: int = 0  # random seed for weights and data

    @property
    def n_ctx(self) -> int:
        """Longest sequence length: half_len * 2."""
        return 2 * self.half_len

    @property
    def d_head(self) -> int:
        """Size of each head's query, key and value vectors: d_model / n_heads."""
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 2 <= self.min_half_len <= self.half_len:
            raise ValueError(
                f"min_half_len ({self.min_half_len}) must be between 2 and half_len ({self.half_len})"
            )


def get_device() -> torch.device:
    """Apple GPU (MPS) if available, otherwise CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
