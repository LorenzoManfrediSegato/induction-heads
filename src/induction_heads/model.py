from induction_heads.config import Config 
import torch
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O= nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool)))

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.pattern = None

        # ablations
        self.ablated_heads = set()

    def forward(self, x):
        batch, seq = x.shape[:2]
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        # split into heads
        q = q.view(batch, seq, self.n_heads, self.d_head)
        k = k.view(batch, seq, self.n_heads, self.d_head)
        v = v.view(batch, seq, self.n_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scores
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        # more masking
        scores = scores.masked_fill(~self.mask[:seq, :seq], float("-inf"))

        # pattern
        pattern = scores.softmax(dim=-1)
        self.pattern = pattern

        out = pattern @ v

        if self.ablated_heads:
            head_mask = torch.ones(self.n_heads, device=out.device) 
            for h in self.ablated_heads:
                head_mask[h] = 0
            out = out * head_mask.view(1, self.n_heads, 1, 1)


        # merging heads back
        out = out.transpose(1, 2).reshape(batch, seq, self.cfg.d_model)

        return self.W_O(out)

class AttnOnlyAutobot(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        self.W_E = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.W_pos = nn.Embedding(cfg.n_ctx, cfg.d_model)

        self.layers = nn.ModuleList([Attention(cfg) for _ in range(cfg.n_layers)])
        self.W_U = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    
    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        resid = self.W_E(tokens) + self.W_pos(positions)

        for layer in self.layers:
            resid = resid + layer(resid)

        return self.W_U(resid)




