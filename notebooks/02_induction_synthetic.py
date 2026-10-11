import torch
import torch.nn.functional as F
from induction_heads.config import Config, get_device
from induction_heads.data import make_repeated_batch, loss_by_half
from induction_heads.model import AttnOnlyAutobot
from induction_heads.plotting import plot_attention_patterns, plot_loss_curves, get_attention_patterns

# setup
cfg = Config(n_steps=5000)
torch.manual_seed(cfg.seed)
device = get_device()
model = AttnOnlyAutobot(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

steps, first, second = [], [], []

# training loop
for step in range(cfg.n_steps):
    tokens = make_repeated_batch(cfg, device=device)
    logits = model(tokens)

    # loss
    predictions = logits[:, :-1]
    targets = tokens[:, 1:]
    predictions = predictions.reshape(-1, cfg.vocab_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(predictions,targets)

    # update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check
    if step % cfg.log_every == 0:
        losses = loss_by_half(logits, tokens, cfg.half_len)
        steps.append(step)
        first.append(losses["first_half"])
        second.append(losses["second_half"])
        print(f"step {step}: first half {losses['first_half']:.3f}, second half {losses['second_half']:.3f}")

plot_loss_curves(steps, first, second, cfg.vocab_size, "figures/induction_loss.png")
with torch.no_grad():
    model(make_repeated_batch(cfg, device=device))
plot_attention_patterns(get_attention_patterns(model), "figures/induction_attention.png", half_len=cfg.half_len)
torch.save(model.state_dict(), "runs/induction_synthetic.pt")

from dataclasses import replace
short_cfg = replace(cfg, half_len=20)
tokens = make_repeated_batch(short_cfg, device=device)
with torch.no_grad():
    logits = model(tokens)
print(loss_by_half(logits, tokens, short_cfg.half_len))

