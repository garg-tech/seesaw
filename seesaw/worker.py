"""Worker process: executes the scheduler's commands on one GPU.

Both workers receive the same commands in the same order, so every collective
lines up. Rank 0 reports decode tokens; whichever rank ends a prefill reports it.
"""
import time
import traceback

import torch

from .engine import Engine, init_worker


def worker_main(rank, port, cmds, results, args):
    dev = init_worker(rank, port)
    try:
        eng = Engine(rank, dev, args["slots"], args["max_len"], args["chunk"], split=args["split"])
        results.put(("ready", rank))
        with torch.inference_mode():
            while True:
                cmd, *a = cmds.get()
                if cmd == "stop":
                    break
                run(eng, rank, cmd, a, results)
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise


def run(eng, rank, cmd, a, results):
    if cmd == "prefill":
        requests, greedy = a
        end_rank = 1 if eng.mode == "pp" else 0
        # Report as each request ends: nothing here waits on the other GPU.
        report = (lambda s, t: results.put(("first", s, t))) if rank == end_rank else None
        eng.prefill(requests, greedy, report)
        torch.cuda.synchronize()
        results.put(("prefill_done", rank))
    elif cmd == "to_tp":
        info = eng.to_tp()
        if rank == 0:
            results.put(("switched", info))
    elif cmd == "to_pp":
        info = eng.to_pp()
        if rank == 0:
            results.put(("switched", info))
    elif cmd == "decode":
        decode(eng, rank, results, *a)
    elif cmd == "release":
        eng.release(a[0])


def decode(eng, rank, results, seqs, greedy):
    """seqs: [(slot, tokens in cache, last token, tokens left)]. Runs until all are done."""
    dev = eng.dev
    slots = [s[0] for s in seqs]
    lens = [s[1] for s in seqs]
    last = torch.tensor([s[2] for s in seqs], device=dev)
    left = [s[3] for s in seqs]
    while slots:
        last = eng.decode_step(slots, lens, last, greedy)
        ids = last.tolist() if rank == 0 else None
        if rank == 0:
            results.put(("tokens", slots, ids))
        keep = [i for i, n in enumerate(left) if n > 1]
        if len(keep) != len(slots):
            slots = [slots[i] for i in keep]
            lens = [lens[i] for i in keep]
            left = [left[i] for i in keep]
            last = last[torch.tensor(keep, dtype=torch.long, device=dev)] if keep else last
        lens = [n + 1 for n in lens]
        left = [n - 1 for n in left]
    if rank == 0:
        results.put(("decode_done",))
