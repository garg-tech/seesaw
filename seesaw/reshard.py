"""Move half tensors between GPUs to switch layouts, PP (split/48-split) <-> TP.

Each layer GPU1 gives up pairs with a same-type layer GPU0 gives up, so GPU0's h1
of one and GPU1's h0 of the other swap in place. Nothing is allocated from the pool
during a switch beyond one tensor of staging.

A split above 24 leaves front layers with no partner. GPU0 holds their h1 in spare
tensors allocated at startup, and GPU1's own copies sit idle under PP.

With park=True (Seesaw, which decodes only in TP) GPU0's spare KV is a small staging
pool instead of a full one. When a sequence finishes prefill, park() sends its extras'
h1 KV straight to GPU1, which needs it for TP anyway, so the switch has none left to move.

Both GPUs keep identical block tables, so block ids mean the same on either side.
"""
import time

import torch
import torch.distributed as dist

STAGE_BLOCKS = 2048     # at most 64 MB of staging per KV exchange
STAGE_BYTES = 64 << 20  # and per weight swap: merged gate_up halves are 118 MB


def swap(t, peer, group):
    """Exchange t with the peer's tensor of the same shape, in place, a piece at a time."""
    for part in t.view(-1).split(STAGE_BYTES // t.element_size()):
        out = part.clone()
        ops = [dist.P2POp(dist.isend, out, peer, group), dist.P2POp(dist.irecv, part, peer, group)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return t.nbytes


def exchange(mine, theirs, blocks, peer, group):
    """Send the listed blocks of `mine`, receive the peer's into `theirs`."""
    n = 0
    if blocks.numel() == 0:
        return n
    # GPU0 is nearly full by now, so stage a piece at a time.
    for idx in blocks.split(STAGE_BLOCKS):
        out = mine.index_select(0, idx)
        got = torch.empty_like(out)
        ops = [dist.P2POp(dist.isend, out, peer, group), dist.P2POp(dist.irecv, got, peer, group)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        theirs.index_copy_(0, idx, got)
        n += out.nbytes
    return n


class Resharder:
    def __init__(self, rank, layers, kv, tables, dev, group, split, park=False):
        self.rank, self.layers, self.kv, self.tables, self.dev, self.group = rank, layers, kv, tables, dev, group
        self.peer = 1 - rank
        idx = lambda ls, full: [l.spec.idx for l in ls if l.spec.full == full]
        front, back = layers[:split], layers[split:]
        self.pairs = [p for full in (False, True) for p in zip(idx(front, full), idx(back, full))]
        for a, b in self.pairs:
            assert layers[a].spec == type(layers[a].spec)(**{**layers[b].spec.__dict__, "idx": a}), (a, b)
        paired = {a for a, _ in self.pairs}
        self.extras = [l.spec.idx for l in front if l.spec.idx not in paired]
        assert not any(layers[a].spec.full for a in self.extras) and len(back) <= len(front), split
        self.parks = park and bool(self.extras)
        self.sending = []
        if rank == 0:
            self.spare_w = {a: {n: torch.empty_like(t) for n, t in layers[a].h[0].items()} for a in self.extras}
            spare = lambda t: (t.new_empty((tables.stage_blocks, *t.shape[1:])) if self.parks
                               else torch.empty_like(t))
            self.spare_k = {a: spare(kv.k[a][0]) for a in self.extras}
            self.spare_v = {a: spare(kv.v[a][0]) for a in self.extras}
        self.spares_loaded = False
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

    def _extras_to_pp(self):
        """Weights never change, so the spares are filled once and GPU1 keeps its copy."""
        n = 0
        for a in self.extras:
            if self.rank == 0:
                w = self.spare_w[a]
                self.layers[a].h[1] = w
                self.kv.k[a][1], self.kv.v[a][1] = self.spare_k[a], self.spare_v[a]
            else:
                w = self.layers[a].h[1]
            if not self.spares_loaded:
                for name in sorted(w):
                    (dist.recv if self.rank == 0 else dist.send)(w[name], self.peer, group=self.group)
                    n += w[name].nbytes
        self.spares_loaded = True
        if self.parks and self.rank == 0:
            self.kv.staged = {(a, 1) for a in self.extras}
            self.tables.start_staging()
        return n

    def park(self, slots):
        """PP prefill: these sequences are done, so their extras' h1 KV goes to GPU1 now.
        Both ranks call it at the same point, so the send lines up behind the hidden states."""
        if not self.parks or not slots:
            return 0
        if self.rank == 0:
            pool, stage = self.tables.unstage(slots)
        else:
            pool = [int(self.tables.sliding[s, b]) for s in slots for b in self.tables.held_sliding(s)]
        if not pool:
            return 0
        if self.rank == 0:
            idx = torch.tensor(stage, device=self.dev)
            # Copied out first, so the next chunk can reuse the staging blocks at once.
            buf = torch.stack([torch.stack([self.kv.k[a][1].index_select(0, idx),
                                            self.kv.v[a][1].index_select(0, idx)]) for a in self.extras])
            self.sending = [x for x in self.sending if not x[0].is_completed()]
            self.sending.append((dist.isend(buf, self.peer, group=self.group), buf))
        else:
            t = self.kv.k[self.extras[0]][1]
            buf = torch.empty((len(self.extras), 2, len(pool), *t.shape[1:]), dtype=t.dtype, device=self.dev)
            dist.recv(buf, self.peer, group=self.group)
            idx = torch.tensor(pool, device=self.dev)
            for i, a in enumerate(self.extras):
                self.kv.k[a][1].index_copy_(0, idx, buf[i, 0])
                self.kv.v[a][1].index_copy_(0, idx, buf[i, 1])
        return buf.nbytes

    def _extras_to_tp(self):
        """GPU0 made these sequences' h1 KV. GPU1 has to end up with it."""
        n = 0
        if self.parks:      # already there
            for req, _ in self.sending:
                req.wait()
            self.sending = []
            if self.rank == 0:
                for a in self.extras:
                    self.layers[a].h.pop(1)
                    self.kv.k[a].pop(1), self.kv.v[a].pop(1)
                self.kv.staged = set()
                self.tables.stop_staging()
            return n
        for a in self.extras:
            if self.rank == 0:
                self.layers[a].h.pop(1)
            for store in (self.kv.k, self.kv.v):
                t = store[a].pop(1) if self.rank == 0 else store[a][1]
                if self.used_sliding.numel() == 0:
                    continue
                for idx in self.used_sliding.split(STAGE_BLOCKS):
                    x = t.index_select(0, idx)
                    if self.rank == 0:
                        dist.send(x, self.peer, group=self.group)
                    else:
                        dist.recv(x, self.peer, group=self.group)
                        t.index_copy_(0, idx, x)
                    n += x.nbytes
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
                n += exchange(store[mine], store[theirs], idx, self.peer, self.group)
            return n
        for store in (k, v):
            if self.rank == 0:
                t = store[a].pop(1)
                n += exchange(t, t, self.used_sliding, self.peer, self.group)
                store[b][0] = t
            else:
                t = store[b].pop(0)
                n += exchange(t, t, self.used_sliding, self.peer, self.group)
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
        torch.cuda.empty_cache()
        dist.barrier(group=self.group)
        t = time.perf_counter()
        self._used()
        w = sum(self._swap_weights(a, b, True) for a, b in self.pairs)
        k = sum(self._swap_kv(a, b) for a, b in self.pairs) + self._extras_to_tp()
        torch.cuda.synchronize()
        self.log.append(("pp->tp", time.perf_counter() - t, w, k))
        return self.log[-1]

    def to_pp(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier(group=self.group)
        t = time.perf_counter()
        w = sum(self._swap_weights(a, b, False) for a, b in self.pairs) + self._extras_to_pp()
        for a, b in self.pairs:
            self._relabel_kv(a, b)
        torch.cuda.synchronize()
        self.log.append(("tp->pp", time.perf_counter() - t, w, 0))
        return self.log[-1]
