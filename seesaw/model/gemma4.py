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
        self.layers = [Layer(s, cfg, ckpt, device, halves) for s in specs]
        self.ropes = {}
        for l in self.layers:
            if l.rope_key not in self.ropes:
                self.ropes[l.rope_key] = rope_table(l.spec.head_dim, cfg.rope[l.rope_key], max_len, device)
        self.kv = KVStore(specs, tables, device, halves)
        self.embed_w = ckpt.full("embed_tokens.weight", device) if embed or head else None
        self.final_norm = ckpt.full("norm.weight", device) if head else None
        self.normalizer = torch.tensor(cfg.hidden ** 0.5, dtype=torch.bfloat16, device=device)

    def embed(self, tokens):
        return F.embedding(tokens, self.embed_w) * self.normalizer

    def forward(self, x, batch, layers=None, comm=None):
        # cos/sin once per pass, not once per layer
        rope = {k: t.index_select(0, batch.positions).unsqueeze(1).chunk(2, -1) for k, t in self.ropes.items()}
        for layer in self.layers if layers is None else layers:
            x = layer.forward(x, batch, self.kv, comm, rope[layer.rope_key])
        return x

    def normed(self, x):
        return rms_norm(x, self.final_norm, self.cfg.eps)

    def logits(self, x):
        y = F.linear(x, self.embed_w).float()
        return torch.tanh(y / self.cfg.softcap) * self.cfg.softcap
