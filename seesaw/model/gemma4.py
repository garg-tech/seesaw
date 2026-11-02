"""The decoder as a stack of half-tensor layers, or any contiguous slice of it."""
import math

import torch
import torch.nn.functional as F

from ..loader import Checkpoint
from .layers import KVStore, Layer, rms_norm, rope_table


class Decoder:
    def __init__(self, cfg, snapshot, device, layer_range, tables, max_len, halves=(0, 1),
                 embed=True, head=True):
        self.cfg, self.device = cfg, device
        ckpt = Checkpoint(snapshot)
        specs = [cfg.layers[i] for i in layer_range]
        ropes = {}
        for s in specs:
            key = "full_attention" if s.full else "sliding_attention"
            if key not in ropes:
                ropes[key] = rope_table(s.head_dim, cfg.rope[key], max_len, device)
        self.layers = [Layer(s, cfg, ckpt, device, ropes["full_attention" if s.full else "sliding_attention"], halves)
                       for s in specs]
        self.kv = KVStore(specs, tables, device, halves)
        self.embed_w = ckpt.full("embed_tokens.weight", device) if embed or head else None
        self.final_norm = ckpt.full("norm.weight", device) if head else None
        self.normalizer = torch.tensor(cfg.hidden ** 0.5, dtype=torch.bfloat16, device=device)

    def embed(self, tokens):
        return F.embedding(tokens, self.embed_w) * self.normalizer

    def forward(self, x, batch, layers, comm=None):
        for layer in layers:
            x = layer.forward(x, batch, self.kv, comm)
        return x

    def normed(self, x):
        return rms_norm(x, self.final_norm, self.cfg.eps)

    def logits(self, x):
        y = F.linear(x, self.embed_w).float()
        return torch.tanh(y / self.cfg.softcap) * self.cfg.softcap
