import torch
from induction_heads.config import Config, get_device
from induction_heads.model import AttnOnlyAutobot
from induction_heads.data import make_repeated_batch
import torch.nn.functional as F

# setup
cfg = Config(n_steps=20000)
device = get_device()
model = AttnOnlyAutobot(cfg).to(device)
L = 20
gen = torch.Generator().manual_seed(0)
tokens = make_repeated_batch(cfg, batch_size=256, half_len=L, generator=gen, device=device)
print(tokens.shape)

state_dict = torch.load('runs/varlen_synthetic.pt', map_location=device)
model.load_state_dict(state_dict)
model.eval()

def losses(tokens, L): 
    with torch.no_grad():
        logits = model(tokens)
    V = logits.size(-1)

    rep = F.cross_entropy(logits[:, L:2*L-1].reshape(-1, V), tokens[:, L+1:2*L].reshape(-1))

    first = F.cross_entropy(logits[:, :L-1].reshape(-1, V), tokens[:, 1:L].reshape(-1))

    return rep.item(), first.item()

layer0 = model.layers[0]

def run(heads, mode="zero"):
    layer0.ablated_heads = set(heads)
    layer0.ablation_mode = mode
    try:
        return losses(tokens, L)
    finally:
        layer0.ablated_heads = set()
        layer0.ablation_mode = "zero"

runs = [
    ("baseline",                   [],           "zero"),
    ("prev-token, mean",           [2, 3],       "mean"),
    ("duplicate-token, mean",      [0, 1],       "mean"),
    ("all layer 0, mean",          [0, 1, 2, 3], "mean"),
    ("cleanup check",              [],           "zero"),
]

for name, heads, mode in runs:
    rep, first = run(heads, mode)
    print(f"{name:28s} repeated {rep:.3f}   first {first:.2f}")
