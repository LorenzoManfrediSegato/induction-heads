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

def run(heads, mode="zero", layer=0):
    attn = model.layers[layer]
    attn.ablated_heads = set(heads)
    attn.ablation_mode = mode
    try:
        return losses(tokens, L)
    finally:
        attn.ablated_heads = set()
        attn.ablation_mode = "zero"

runs = [
    ("baseline",               [],           "zero", 0),
    ("prev-token, mean",       [2, 3],       "mean", 0),
    ("duplicate-token, mean",  [0, 1],       "mean", 0),
    ("all layer 0, mean",      [0, 1, 2, 3], "mean", 0),
    ("all layer 1, zero",      [0, 1, 2, 3], "zero", 1),
    ("all layer 1, mean",      [0, 1, 2, 3], "mean", 1),
    ("cleanup check",          [],           "zero", 0),  
    ("prev-token, zero",       [2, 3],       "zero", 0),
    ("duplicate-token, zero",  [0, 1],       "zero", 0),
    ("all layer 0, zero",      [0, 1, 2, 3], "zero", 0),
]

for name, heads, mode, layer in runs:
    rep, first = run(heads, mode, layer)
    print(f"{name:28s} repeated {rep:.3f}   first {first:.2f}")
