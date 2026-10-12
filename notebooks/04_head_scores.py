import torch
from induction_heads.config import Config, get_device
from induction_heads.model import AttnOnlyAutobot
from induction_heads.data import make_repeated_batch
from induction_heads.scores import score

# setup
cfg = Config(n_steps=20000)
device = get_device()
model = AttnOnlyAutobot(cfg).to(device)

# load trained weights
model.load_state_dict(torch.load("runs/varlen_synthetic.pt", map_location=device))
model.eval()

for L in [10, 20, 32]:
    print(f"--- repeat length {L} ---")
    # scoring batch
    g = torch.Generator().manual_seed(0)
    tokens = make_repeated_batch(cfg, batch_size=64, half_len=L, generator=g, device=device)

    # one forward pass
    with torch.no_grad():
        model(tokens)

    # read patterns
    patterns = [layer.pattern for layer in model.layers]


    prev_token = torch.arange(1, 2 * L, device=device)
    induction = torch.arange(L, 2 * L, device=device)
    dupe_token = torch.arange(L, 2 * L, device = device)

    for layer_idx, pattern in enumerate(patterns):
        prev_token_score = score(pattern, prev_token, 1)
        induction_score = score(pattern, induction, L - 1)
        dupe_token_score = score(pattern, dupe_token, L)

        for head in range(cfg.n_heads):
            print(layer_idx, head,
                round(prev_token_score[head].item(), 3),
                round(induction_score[head].item(), 3),
                round(dupe_token_score[head].item(), 3))

