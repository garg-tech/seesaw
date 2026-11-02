"""Move half tensors between GPUs to switch layouts, PP (24/24) <-> TP.

The split is even and the sliding/full pattern repeats every 6 layers, so layer l
and layer l + 24 have the same shapes. GPU0's h1 of layer l and GPU1's h0 of layer
l + 24 therefore swap in place: no spare memory beyond one tensor of staging, and
nothing is allocated from the pool during a switch.

Both GPUs keep identical block tables, so block ids mean the same on either side.
"""
import time

import torch
import torch.distributed as dist

SPLIT = 24


def swap(t, peer, group):
    """Exchange t with the peer's tensor of the same shape, in place."""
    out = t.clone()
    ops = [dist.P2POp(dist.isend, out, peer, group), dist.P2POp(dist.irecv, t, peer, group)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return t.nbytes


def swap_blocks(t, blocks, peer, group):
    """Same, for the listed blocks of a KV tensor only."""
    if blocks.numel() == 0:
        return 0
    out = t.index_select(0, blocks)
    got = torch.empty_like(out)
    ops = [dist.P2POp(dist.isend, out, peer, group), dist.P2POp(dist.irecv, got, peer, group)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    t.index_copy_(0, blocks, got)
    return out.nbytes


class Resharder:
    def __init__(self, rank, layers, kv, tables, dev, group):
        self.rank, self.layers, self.kv, self.tables, self.dev, self.group = rank, layers, kv, tables, dev, group
        self.peer = 1 - rank
        self.pairs = [(l, l + SPLIT) for l in range(SPLIT)]
        for a, b in self.pairs:
            assert layers[a].spec == type(layers[a].spec)(**{**layers[b].spec.__dict__, "idx": a}), (a, b)
        self.log = []

    def _swap_weights(self, a, b, to_tp):
        """On GPU0 the moving half belongs to layer a, on GPU1 to layer b."""
        L = self.layers
        n = 0
        if self.rank == 0:
            src, dst, hs, hd = (L[a], L[b], 1, 0) if to_tp else (L[b], L[a], 0, 1)
        else:
            src, dst, hs, hd = (L[b], L[a], 0, 1) if to_tp else (L[a], L[b], 1, 0)
        w = src.h.pop(hs)
        for name in sorted(w):
            n += swap(w[name], self.peer, self.group)
        dst.h[hd] = w
        return n

    def _swap_kv(self, a, b):
        """PP -> TP only. Sequences in flight keep their blocks."""
        spec = self.layers[a].spec
        k, v, n = self.kv.k, self.kv.v, 0
        if spec.full:
            # One KV head can't split, so each GPU gets a copy of the other's.
            idx = self.used_full
            mine, theirs = (a, b) if self.rank == 0 else (b, a)
            for store in (k, v):
                out = store[mine].index_select(0, idx)
                got = torch.empty_like(out)
                ops = [dist.P2POp(dist.isend, out, self.peer, self.group),
                       dist.P2POp(dist.irecv, got, self.peer, self.group)]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
                store[theirs].index_copy_(0, idx, got)
                n += out.nbytes
            return n
        for store in (k, v):
            if self.rank == 0:
                t = store[a].pop(1)
                n += swap_blocks(t, self.used_sliding, self.peer, self.group)
                store[b][0] = t
            else:
                t = store[b].pop(0)
                n += swap_blocks(t, self.used_sliding, self.peer, self.group)
                store[a][1] = t
        return n

    def _relabel_kv(self, a, b):
        """TP -> PP. The cache is empty, so the tensors only change owner."""
        if self.layers[a].spec.full:
            return
        for store in (self.kv.k, self.kv.v):
            if self.rank == 0:
                store[a][1] = store[b].pop(0)
            else:
                store[b][0] = store[a].pop(1)

    def _used(self):
        self.used_sliding = torch.tensor(self.tables.used_sliding(), device=self.dev)
        self.used_full = torch.tensor(self.tables.used_full(), device=self.dev)

    def to_tp(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        t = time.perf_counter()
        self._used()
        w = sum(self._swap_weights(a, b, True) for a, b in self.pairs)
        k = sum(self._swap_kv(a, b) for a, b in self.pairs)
        torch.cuda.synchronize()
        self.log.append(("pp->tp", time.perf_counter() - t, w, k))
        return self.log[-1]

    def to_pp(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        t = time.perf_counter()
        w = sum(self._swap_weights(a, b, False) for a, b in self.pairs)
        for a, b in self.pairs:
            self._relabel_kv(a, b)
        torch.cuda.synchronize()
        self.log.append(("tp->pp", time.perf_counter() - t, w, 0))
        return self.log[-1]
