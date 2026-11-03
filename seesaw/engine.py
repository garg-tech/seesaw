"""Per-GPU engine: prefill and decode on one worker process.

Rank 0 is the A5000, rank 1 the PRO 4000. Each worker sees only its own GPU, so
Triton compiles for the right architecture.
"""
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from .config import ModelConfig, snapshot_dir
from .decode_graph import DecodeGraphs
from .kv_cache import Batch, BlockTables
from .model.gemma4 import Decoder
from .reshard import Resharder

# Sockets the GPUs hang off, from lscpu / nvidia-smi topo.
NUMA_CPUS = {0: [*range(0, 16), *range(32, 48)], 1: [*range(16, 32), *range(48, 64)]}
TOP_K = 64
TOP_P = 0.95


def init_worker(rank, port):
    """Call first in a worker, before anything touches CUDA."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    os.sched_setaffinity(0, NUMA_CPUS[rank])
    dev = torch.device("cuda", 0)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=dev)
    return dev


class Engine:
    def __init__(self, rank, dev, slots, max_len, chunk, mode="tp", scratch_chunks=2, graphs=True, split=24):
        self.rank, self.dev, self.chunk, self.mode = rank, dev, chunk, "tp"
        self.cfg = ModelConfig.load()
        self.tables = BlockTables(slots, max_len, self.cfg.window,
                                  scratch_blocks=scratch_chunks * (chunk // 16 + 1))
        self.comm = dist.group.WORLD
        # Runs on the compute stream. torch's collectives use a second stream and
        # sync to it by event on every call, too slow for 96 all-reduces a step.
        self.tp_comm = PyNcclCommunicator(dist.new_group(backend="gloo"), dev)
        self.dec = Decoder(self.cfg, snapshot_dir(), dev, range(len(self.cfg.layers)),
                           self.tables, max_len, halves=(rank,))
        self.split = split
        self.resharder = Resharder(rank, self.dec.layers, self.dec.kv, self.tables, dev, self.comm, split)
        self.active = self.dec.layers
        self._sending = None
        half = self.cfg.vocab // 2
        self.vocab_start = rank * half
        self.head_w = self.dec.embed_w[self.vocab_start:self.vocab_start + half]
        self.gen = torch.Generator(dev).manual_seed(0)
        self.graphs = None
        if graphs:
            with torch.inference_mode():
                self.graphs = DecodeGraphs(self.tables, dev, slots, self._graph_step)
        if mode == "pp":
            self.to_pp()

    # ---------------------------------------------------------------- switching
    def to_pp(self):
        out = self.resharder.to_pp()
        self.mode = "pp"
        self.active = self.dec.layers[:self.split] if self.rank == 0 else self.dec.layers[self.split:]
        return out

    def to_tp(self):
        if self._sending is not None:
            self._sending.wait()
            self._sending = None
        out = self.resharder.to_tp()
        self.mode = "tp"
        self.active = self.dec.layers
        return out

    # ---------------------------------------------------------------- sampling
    def _scores(self, x):
        w = self.dec.embed_w if self.mode == "pp" else self.head_w
        return torch.nn.functional.linear(self.dec.normed(x), w).float()

    def logprobs(self, x):
        """Full-vocab log-probabilities, the way vLLM reports them. For tests."""
        y = self._scores(x)
        if self.mode == "tp":
            out = [torch.empty_like(y) for _ in range(2)]
            dist.all_gather(out, y, group=self.comm)
            y = torch.cat(out, -1)
        return (torch.tanh(y / self.cfg.softcap) * self.cfg.softcap).log_softmax(-1)

    def _candidates(self, x, k):
        """Top-k over the whole vocab. Under TP each GPU scores half, then they swap."""
        vals, ids = self._scores(x).topk(k, dim=-1)
        if self.mode == "pp":
            return vals, ids
        ids = ids + self.vocab_start
        out_v, out_i = [torch.empty_like(vals) for _ in range(2)], [torch.empty_like(ids) for _ in range(2)]
        dist.all_gather(out_v, vals, group=self.comm)
        dist.all_gather(out_i, ids, group=self.comm)
        return torch.cat(out_v, -1), torch.cat(out_i, -1)

    def _choose(self, vals, ids, greedy):
        if greedy:
            return ids.gather(1, vals.argmax(-1, keepdim=True)).squeeze(1)
        cap = self.cfg.softcap
        vals, order = (torch.tanh(vals / cap) * cap).topk(TOP_K, dim=-1)
        p = vals.softmax(-1)
        keep = (p.cumsum(-1) - p) < TOP_P
        pick = torch.multinomial(p * keep, 1, generator=self.gen)
        return ids.gather(1, order.gather(1, pick)).squeeze(1)

    def sample(self, x, greedy, n):
        """Next token for n rows of x. Under PP rank 0 has no x, only n."""
        root = 1 if self.mode == "pp" else 0
        have = x is not None
        if have:
            vals, ids = self._candidates(x, 1 if greedy else TOP_K)
        if self.rank == root:
            tok = self._choose(vals, ids, greedy)
        else:
            tok = torch.empty(n, dtype=torch.int64, device=self.dev)
        dist.broadcast(tok, root, group=self.comm)
        return tok

    # ---------------------------------------------------------------- forward
    def _graph_step(self, inputs):
        return self.dec.forward(self.dec.embed(inputs.tokens), inputs, self.dec.layers, self.tp_comm)

    def _forward(self, tokens, b):
        """Run this GPU's part of a pass. Returns hidden states on the GPU that ends it."""
        if self.mode == "tp":
            return self.dec.forward(self.dec.embed(tokens), b, self.active, self.tp_comm)
        if self.rank == 0:
            x = self.dec.forward(self.dec.embed(tokens), b, self.active)
            # Only one pass waits to cross, so stage 0 stays one pass ahead of stage 1.
            if self._sending is not None:
                self._sending.wait()
            self._sending = dist.isend(x, 1, group=self.comm)
            self._in_flight = x
            return None
        x = torch.empty((tokens.shape[0], self.cfg.hidden), dtype=torch.bfloat16, device=self.dev)
        dist.recv(x, 0, group=self.comm)
        return self.dec.forward(x, b, self.active)

    def prefill(self, requests, greedy=False, report=None):
        """requests: [(slot, token list)]. Chunks every prompt.

        Under PP only rank 1 sees the outputs. It calls report(slot, token) as each
        request ends; stage 0 doesn't wait for that, or the pipeline would stall.
        """
        first = {}
        pending = [[s, 0, np.asarray(t)] for s, t in requests]
        while pending:
            slots, starts, lens, toks, room = [], [], [], [], self.chunk
            for s, done, t in pending:
                n = min(len(t) - done, room)
                if n == 0:
                    break
                self.tables.extend(s, done + n)
                slots.append(s); starts.append(done); lens.append(n); toks.append(t[done:done + n])
                room -= n
            b = Batch(self.tables, slots, starts, lens, self.dev)
            y = self._forward(torch.from_numpy(np.concatenate(toks)).to(self.dev), b)
            finished = [i for i in range(len(slots)) if starts[i] + lens[i] == len(pending[i][2])]
            if finished:
                ends = b.cu_seqlens[1:].long() - 1
                tok = None
                if self.mode == "tp":
                    tok = self.sample(y[ends[finished]], greedy, len(finished))
                elif self.rank == 1:
                    vals, ids = self._candidates(y[ends[finished]], 1 if greedy else TOP_K)
                    tok = self._choose(vals, ids, greedy)
                if tok is not None:
                    for i, t in zip(finished, tok.tolist()):
                        first[slots[i]] = t
                        if report:
                            report(slots[i], t)
            for s, st, n in zip(slots, starts, lens):
                self.tables.trim(s, st + n)
            for i, n in enumerate(lens):
                pending[i][1] += n
            pending = [p for p in pending if p[1] < len(p[2])]
        return first

    def decode_step(self, slots, lens, tokens, greedy=False):
        """One token per slot. lens: tokens already in cache; tokens: LongTensor of last outputs."""
        for s, n in zip(slots, lens):
            self.tables.extend(s, n + 1)
        if self.graphs and self.mode == "tp":
            y = self.graphs.step(slots, lens, tokens)
        else:
            y = self._forward(tokens, Batch(self.tables, slots, lens, [1] * len(slots), self.dev))
        for s, n in zip(slots, lens):
            self.tables.trim(s, n + 1)
        return self.sample(y, greedy, len(slots))

    def decode(self, slots, lens, tokens, steps, greedy=False, on_step=None):
        lens = list(lens)
        for i in range(steps):
            tokens = self.decode_step(slots, lens, tokens, greedy)
            lens = [n + 1 for n in lens]
            if on_step:
                on_step(i, tokens)
        return tokens, lens

    def release(self, slots):
        for s in slots:
            self.tables.release(s)
