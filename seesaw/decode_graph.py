"""Decode steps replayed from CUDA graphs, the way vLLM runs them.

Python and op dispatch cost about 35 ms a step, whatever the batch, because a step is
48 layers of 90 ops. A graph replays them without the host. It holds fixed addresses,
so each batch size gets its own input tensors, and smaller batches pad up to the next
size. A padded row sees one key, writes nothing (slot -1) and its output is dropped.

A re-shard swaps contents in place and a round trip returns every buffer to the same
layer and half, so graphs captured in TP stay valid after any number of switches.
"""
import numpy as np
import torch
from vllm.utils.torch_utils import current_stream

from .config import BLOCK_SIZE

SIZES = (1, 2, 4, 8, 16, 24, 32)


class Inputs:
    """Stands in for a Batch: same fields, at addresses that never change."""

    def __init__(self, size, tables, dev):
        self.size = size
        self.tokens = torch.zeros(size, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(size, dtype=torch.int64, device=dev)
        self.cu_seqlens = torch.arange(size + 1, dtype=torch.int32, device=dev)
        self.seq_lens = torch.ones(size, dtype=torch.int32, device=dev)
        self.table_full = torch.zeros((size, tables.max_blocks), dtype=torch.int32, device=dev)
        self.table_sliding = torch.zeros((size, tables.max_blocks), dtype=torch.int32, device=dev)
        self.map_full = torch.full((size,), -1, dtype=torch.int64, device=dev)
        self.map_sliding = torch.full((size,), -1, dtype=torch.int64, device=dev)
        self.max_q = 1
        self.max_k = tables.max_blocks * BLOCK_SIZE

    def fill(self, tables, slots, lens, tokens):
        n = len(slots)
        s, pos = np.asarray(slots), np.asarray(lens, np.int64)
        blk, off = pos // BLOCK_SIZE, pos % BLOCK_SIZE
        positions = np.zeros(self.size, np.int64)
        seq_lens = np.ones(self.size, np.int32)
        table_full = np.zeros((self.size, tables.max_blocks), np.int32)
        table_sliding = np.zeros((self.size, tables.max_blocks), np.int32)
        map_full = np.full(self.size, -1, np.int64)
        map_sliding = np.full(self.size, -1, np.int64)
        positions[:n] = pos
        seq_lens[:n] = pos + 1
        table_full[:n] = tables.full[s]
        table_sliding[:n] = tables.sliding[s]
        map_full[:n] = tables.full[s, blk].astype(np.int64) * BLOCK_SIZE + off
        map_sliding[:n] = tables.sliding[s, blk].astype(np.int64) * BLOCK_SIZE + off
        for t, a in ((self.positions, positions), (self.seq_lens, seq_lens),
                     (self.table_full, table_full), (self.table_sliding, table_sliding),
                     (self.map_full, map_full), (self.map_sliding, map_sliding)):
            t.copy_(torch.from_numpy(a))
        self.tokens[:n] = tokens
        self.tokens[n:] = 0


class DecodeGraphs:
    def __init__(self, tables, dev, slots, step):
        """step(inputs) runs one TP decode pass and returns hidden states."""
        self.tables = tables
        self.sizes = sorted({min(s, slots) for s in SIZES})
        self.graphs = {}
        pool = None
        # Largest first, so the smaller graphs fit in the pool it leaves.
        for size in reversed(self.sizes):
            inputs = Inputs(size, tables, dev)
            step(inputs)    # a kernel's first launch loads it, which capture can't record
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=current_stream()):
                out = step(inputs)
            pool = graph.pool() if pool is None else pool
            self.graphs[size] = (graph, inputs, out)

    def step(self, slots, lens, tokens):
        n = len(slots)
        size = next(s for s in self.sizes if s >= n)
        graph, inputs, out = self.graphs[size]
        inputs.fill(self.tables, slots, lens, tokens)
        graph.replay()
        return out[:n]
