"""Where a decode step's time goes.

  python bench/profile_decode.py [--batches 1 32] [--prompt 512] [--steps 48]

Wall time on rank 0 around the kernels, the collectives and the sampler. Collectives
return before the GPU finishes, so a row is host time, not GPU time; `sync` at the end
of a step is what the host still had to wait for.
"""
import argparse
import collections
import socket
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")


def worker(rank, port, a):
    import torch.distributed as dist

    import seesaw.engine as E
    import seesaw.model.layers as L
    dev = E.init_worker(rank, port)
    eng = E.Engine(rank, dev, slots=max(a.batches), max_len=a.prompt + a.steps + 64, chunk=2048)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    spent, calls = collections.Counter(), collections.Counter()

    def timed(name, fn):
        def wrap(*args, **kw):
            t = time.perf_counter()
            r = fn(*args, **kw)
            spent[name] += time.perf_counter() - t
            calls[name] += 1
            return r
        return wrap

    L.unified_attention = timed("attention kernel", L.unified_attention)
    L.triton_reshape_and_cache_flash = timed("kv write kernel", L.triton_reshape_and_cache_flash)
    eng.tp_comm.all_reduce = timed("all_reduce", eng.tp_comm.all_reduce)
    dist.all_gather = timed("all_gather", dist.all_gather)
    dist.broadcast = timed("broadcast", dist.broadcast)
    eng._forward = timed("forward (total)", eng._forward)
    eng.sample = timed("sample (total)", eng.sample)
    eng._scores = timed("scores", eng._scores)
    E.Batch = timed("Batch", E.Batch)

    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for B in a.batches:
            reqs = [(s, rng.integers(1000, 200000, a.prompt).tolist()) for s in range(B)]
            first = eng.prefill(reqs, greedy=a.greedy)
            slots = list(range(B))
            toks = torch.tensor([first[s] for s in slots], device=dev)
            lens = [a.prompt] * B
            eng.decode(slots, lens, toks, 8, a.greedy)
            lens = [n + 8 for n in lens]
            spent.clear(); calls.clear()
            torch.cuda.synchronize()
            n, total, waited = a.steps - 8, time.perf_counter(), 0.0
            for _ in range(n):
                toks = eng.decode_step(slots, lens, toks, a.greedy)
                t = time.perf_counter()
                torch.cuda.synchronize()
                waited += time.perf_counter() - t
                lens = [x + 1 for x in lens]
            total = time.perf_counter() - total
            eng.release(slots)
            log(f"\nB={B}: {total / n * 1e3:.1f} ms/step, of which end-of-step sync {waited / n * 1e3:.2f} ms")
            for name, s in spent.most_common():
                log(f"  {name:20} {s / n * 1e3:7.2f} ms/step  {calls[name] / n:6.1f} calls  "
                    f"{s / max(calls[name], 1) * 1e6:7.0f} us each")
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, a), nprocs=2)
