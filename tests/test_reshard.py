"""PP prefill -> re-shard -> TP decode must match vLLM, then go back and do it again.

  python tests/test_reshard.py [split]

Uses the golden cases, teacher-forced like test_tp_golden.py. Three sequences are
in flight at once so block ids and slot offsets get exercised, and the second
round proves TP -> PP restored the weights.
"""
import json
import socket
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")


def check(eng, cases, dev, log):
    idx = [3, 4, 5]
    reqs = [(s, cases[i]["prompt"]) for s, i in enumerate(idx)]
    eng.prefill(reqs, greedy=True)
    info = eng.to_tp()
    log(f"  {info[0]}: {info[1]:.2f} s, weights {info[2] / 2**30:.2f} GB, kv {info[3] / 2**30:.2f} GB")
    slots = [s for s, _ in reqs]
    n = [len(p) for _, p in reqs]
    errs = []
    gold = [cases[i]["tokens"] for i in idx]
    for j in range(len(gold[0]) - 1):
        toks = torch.tensor([g[j] for g in gold], device=dev)
        for s, m in zip(slots, n):
            eng.tables.extend(s, m + 1)
        from seesaw.kv_cache import Batch
        b = Batch(eng.tables, slots, n, [1] * len(slots), dev)
        y = eng._forward(toks, b)
        for s, m in zip(slots, n):
            eng.tables.trim(s, m + 1)
        n = [m + 1 for m in n]
        lp = eng.logprobs(y)
        if eng.rank == 0:
            errs.append([abs(lp[r, gold[r][j + 1]].item() - cases[idx[r]]["logprobs"][j + 1]) for r in range(3)])
    eng.release(slots)
    return np.array(errs)


def worker(rank, port, cases, split):
    from seesaw.engine import Engine, init_worker
    dev = init_worker(rank, port)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    max_len = 6144 + 96
    eng = Engine(rank, dev, slots=4, max_len=max_len, chunk=2048, mode="pp", split=split)
    log(f"startup {eng.resharder.log[-1]}")
    bad = 0
    with torch.inference_mode():
        for round in range(2):
            log(f"round {round}")
            e = check(eng, cases, dev, log)
            if rank == 0:
                log(f"  logprob err mean per case {e.mean(0).round(3).tolist()} max {e.max():.3f}")
                bad += e.mean() > 0.15
            info = eng.to_pp()
            log(f"  {info[0]}: {info[1]:.2f} s, weights {info[2] / 2**30:.2f} GB")
    log("FAIL" if bad else "PASS")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with open("tests/golden/golden.json") as f:
        cases = json.load(f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, cases, int(sys.argv[1]) if len(sys.argv) > 1 else 24), nprocs=2)
