"""Measure t_step(B) and prefill speed of the engine (PLAN.md Gates 1 and 2).

  python bench/engine_micro.py [--mode tp|pp] [--batches 1 4 16 32] [--chunk 2048]
"""
import argparse
import socket
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")


def worker(rank, port, a):
    from seesaw.engine import Engine, init_worker
    dev = init_worker(rank, port)
    t0 = time.perf_counter()
    eng = Engine(rank, dev, slots=max(a.batches), max_len=a.prompt + a.steps + 64, chunk=a.chunk, mode=a.mode)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    log(f"loaded in {time.perf_counter() - t0:.0f}s, gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for B in a.batches:
            reqs = [(s, rng.integers(1000, 200000, a.prompt).tolist()) for s in range(B)]
            torch.cuda.synchronize(); t = time.perf_counter()
            first = eng.prefill(reqs)
            torch.cuda.synchronize(); pre = time.perf_counter() - t
            if a.mode == "pp":      # Seesaw decodes in TP
                eng.release(list(range(B)))
                log(f"B={B:<3} prefill {B * a.prompt / pre:6.0f} tok/s ({pre / B:.2f} s/req)")
                continue
            slots = list(range(B))
            toks = torch.tensor([first[s] for s in slots], device=dev)
            lens = [a.prompt] * B
            eng.decode(slots, lens, toks, 8)      # warm the kernels
            lens = [n + 8 for n in lens]
            torch.cuda.synchronize(); t = time.perf_counter()
            eng.decode(slots, lens, toks, a.steps - 8)
            torch.cuda.synchronize(); dec = (time.perf_counter() - t) / (a.steps - 8)
            eng.release(slots)
            log(f"B={B:<3} prefill {B * a.prompt / pre:6.0f} tok/s ({pre / B:.2f} s/req)   "
                f"t_step {dec * 1e3:6.1f} ms  ({dec * 1e3 / B:5.2f} ms/tok, {B / dec:5.0f} tok/s)")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--prompt", type=int, default=6144)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--mode", default="tp")
    a = p.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, a), nprocs=2)
