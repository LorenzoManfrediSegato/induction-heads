## Model configuration

The model is trained on synthetic data: each sequence is some random tokens followed by an exact copy of them, e.g. `[17, 3, 99, 42, 17, 3, 99, 42]`. The first half is unpredictable; the second half can be predicted perfectly by a model that has learned to copy, which is what induction heads do.

The length of the random half changes every batch (between `min_half_len` and `half_len`). A first version used a fixed length of 32, and the model learned a positional shortcut instead of induction: "at position i, copy whatever is at position i − 31". It reached near-perfect loss (0.004) on the repeated half, but on sequences that repeat after 20 tokens its loss was 11.4, worse than random guessing (6.24). Varying the length makes position useless for finding the earlier copy, so the model has to match tokens by content.

### Architecture

| Name | Value | What it is | Why this value |
|---|---|---|---|
| `vocab_size` | **512** | Number of distinct tokens | A random first half of 32 tokens has 32 × 31 ÷ 2 = 496 position pairs, each matching by chance with probability 1 ÷ `vocab_size`. With 512 tokens that's ~1 repeated pair per sequence (vs ~3.9 with 128). Fewer repeats mean fewer cases where "what came next last time" is ambiguous |
| `half_len` | **32** | Longest random first half | Gives the induction head up to 32 earlier positions to look back over |
| `min_half_len` | **8** | Shortest random first half | Each batch picks a length from 8 to 32, so the distance to the earlier copy varies and a positional shortcut can't work |
| `n_ctx` | **64** | Longest sequence length (`half_len` 32 × 2) | Follows from `half_len`; shorter batches use the first 2 × length positions |
| `n_layers` | **2** | Attention layers | Induction needs two steps: a layer-0 head moves "previous token" information, a layer-1 head uses it to find the earlier copy |
| `d_model` | **64** | Width of the residual stream (the shared vector every layer reads from and writes to) | Small enough to train in minutes; enough room for token identity plus position |
| `n_heads` | **4** per layer | Attention heads in each layer | Room for a previous-token head and an induction head to form, while keeping the attention patterns easy to inspect (4 heads × 2 layers = 8) |
| `d_head` | **16** | Size of each head's query, key and value vectors (`d_model` 64 ÷ `n_heads` 4) | Standard split: the heads together cover the full width |
| MLPs | **none** | — | Attention-only, as in *A Mathematical Framework for Transformer Circuits* |
| LayerNorm | **none** | A normalisation step between layers | Keeps the model close to the paper's simplified setting and easier to interpret |
| Positions | **learned `W_pos`** (`n_ctx` 64 × `d_model` 64) | One learned vector per position, added to the token embedding | Simplest positional scheme |

Total: ~102k parameters — `W_E` (512 × 64 = 32,768) + `W_pos` (64 × 64 = 4,096) + 2 layers × Q, K, V, O (32,768) + `W_U` (64 × 512 = 32,768).

### Training

| Name | Value | What it is | Why |
|---|---|---|---|
| `batch_size` | **64** | Sequences per training step | Small and fast on a laptop GPU (Apple M5, MPS) |
| `lr` | **1e-3** | Learning rate (how big each weight update is) | Standard for a model this size with Adam |
| Optimizer | **AdamW**, `weight_decay` 0.01 | The update rule | Standard choice |
| `n_steps` | **20,000** | Training steps | Induction heads typically form well before this |
| Data | **fresh random batch every step** | — | No fixed dataset, so the model can't memorise — it has to learn the general copying rule |

### What to expect

- **Step-0 loss:** ≈ ln(`vocab_size` 512) ≈ 6.24 on both halves (uniform guessing).
- **First-half loss:** stays near 6.24 throughout — random tokens can't be predicted.
- **Second-half loss:** falls towards ~0, typically after a plateau followed by a sudden drop — the induction head forming. Position 32 (first token of the copy) stays unpredictable.
