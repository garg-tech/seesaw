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
        # PP decode also sends tokens back 1 -> 0 on it: a second NCCL group would cost
        # GPU0 buffer memory it doesn't have at 26/22.
        self.tp_comm = PyNcclCommunicator(dist.new_group(backend="gloo"), dev)
        self.side = torch.cuda.Stream(dev)
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

    # ---------------------------------------------------------------- PP decode
    def capture_pp_graphs(self):
        """Per-stage decode graphs, one set per micro-batch: one group's output can still be
        crossing to stage 1 while the other group runs."""
        if self.rank == 0:
            step, hidden = (lambda i: self.dec.forward(self.dec.embed(i.tokens), i, self.active)), None
        else:
            step, hidden = (lambda i: self.dec.forward(i.hidden, i, self.active)), self.cfg.hidden
        half = (self.tables.slots + 1) // 2
        with torch.inference_mode():
            self.pp_graphs = [DecodeGraphs(self.tables, self.dev, half, step, hidden) for _ in range(2)]

    def _stage(self, g, k):
        """This GPU's layers for group g, k = 0/1. Stage 1 receives its input first."""
        slots, lens = g["slots"], g["lens"]
        for s, n in zip(slots, lens):
            self.tables.extend(s, n + 1)
        graphs = getattr(self, "pp_graphs", None)
        if self.rank == 0:
            if graphs:
                y = graphs[k].step(slots, lens, g["tok"])
            else:
                y = self.dec.forward(self.dec.embed(g["tok"]), Batch(self.tables, slots, lens, [1] * len(slots), self.dev), self.active)
        else:
            recv = lambda x: dist.recv(x, 0, group=self.comm)
            if graphs:
                y = graphs[k].step(slots, lens, None, lambda i: recv(i.hidden[:len(slots)]))
            else:
                x = torch.empty((len(slots), self.cfg.hidden), dtype=torch.bfloat16, device=self.dev)
                recv(x)
                y = self.dec.forward(x, Batch(self.tables, slots, lens, [1] * len(slots), self.dev), self.active)
        for s, n in zip(slots, lens):
            self.tables.trim(s, n + 1)
        return y

    def decode_pp(self, slots, lens, tokens, left, greedy=False, report=None, force=None):
        """Decode under PP the way vLLM runs it: two micro-batches in flight, so stage 0 runs
        one group while stage 1 runs the other. Stage 1 samples, sends the tokens back to
        stage 0 and calls report(slots, ids). force: {slot: tokens} feeds those back instead
        of the sampled ones, for teacher-forced tests."""
        cut = (len(slots) + 1) // 2
        groups = [dict(k=k, slots=list(slots[a:b]), lens=list(lens[a:b]), left=list(left[a:b]),
                       tok=tokens[a:b], step=0)
                  for k, (a, b) in enumerate(((0, cut), (cut, len(slots)))) if b > a]
        done = []      # stage 1: (slots, pinned ids, event) not yet reported
        while groups:
            for g in groups:
                if self.rank == 0:
                    if "recv" in g:             # waits on the stream, not the host
                        torch.cuda.current_stream().wait_event(g.pop("recv"))
                        g["tok"] = g.pop("tokbuf")[g.pop("keep")]
                    if "send" in g:             # the graph's output is about to be overwritten
                        g.pop("send").wait()
                y = self._stage(g, g["k"])
                more = any(n > 1 for n in g["left"])
                keep = [i for i, n in enumerate(g["left"]) if n > 1]
                if self.rank == 0:
                    g["send"] = dist.isend(y, 1, group=self.comm)
                    if more:
                        g["tokbuf"] = torch.empty(len(g["slots"]), dtype=torch.int64, device=self.dev)
                        g["keep"] = torch.tensor(keep, dtype=torch.long, device=self.dev)
                        # On a side stream, so this GPU runs the other group while it waits.
                        self.side.wait_stream(torch.cuda.current_stream())
                        self.tp_comm.recv(g["tokbuf"], 1, stream=self.side)
                        g["tokbuf"].record_stream(self.side)
                        g["recv"] = torch.cuda.Event()
                        g["recv"].record(self.side)
                else:
                    vals, ids = self._candidates(y, 1 if greedy else TOP_K)
                    tok = out = self._choose(vals, ids, greedy)
                    if force is not None and more:
                        tok = torch.tensor([force[s][g["step"] + 1] for s in g["slots"]], device=self.dev)
                    if more:
                        self.tp_comm.send(tok, 0)
                    host = torch.empty(len(out), dtype=torch.int64, pin_memory=True)
                    host.copy_(out, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record()
                    done.append((list(g["slots"]), host, ev))
                    while done and done[0][2].query():
                        self._report(report, *done.pop(0)[:2])
                g["step"] += 1
                for key in ("slots", "lens", "left"):
                    g[key] = [g[key][i] for i in keep]
                g["lens"] = [n + 1 for n in g["lens"]]
                g["left"] = [n - 1 for n in g["left"]]
            groups = [g for g in groups if g["slots"]]
        torch.cuda.synchronize()
        for s, host, _ in done:
            self._report(report, s, host)

    @staticmethod
    def _report(report, slots, host):
        if report:
            report(slots, host.tolist())

    def release(self, slots):
        for s in slots:
            self.tables.release(s)
