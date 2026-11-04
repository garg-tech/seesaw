"""Pipelined PP decode (two micro-batches, per-stage graphs) must reproduce vLLM.

  python tests/test_pp_decode.py [split]

Teacher-forced like test_tp_golden.py. Four sequences in flight make two groups of
two, so both micro-batch graph sets and the token return path are exercised.
"""
import json
import socket
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")
IDX = [2, 3, 4, 5]


def worker(rank, port, cases, split):
    from seesaw.engine import Engine, init_worker
    dev = init_worker(rank, port)
    gold = [cases[i]["tokens"] for i in IDX]
    max_len = max(len(cases[i]["prompt"]) for i in IDX) + len(gold[0]) + 16
    eng = Engine(rank, dev, slots=len(IDX), max_len=max_len, chunk=2048, mode="pp", graphs=False, split=split)
    eng.capture_pp_graphs()
    rows = []
    orig = eng._candidates

    def spy(x, k):
        rows.append(eng.logprobs(x))
        return orig(x, k)

    slots = list(range(len(IDX)))
    with torch.inference_mode():
        eng.prefill([(s, cases[i]["prompt"]) for s, i in zip(slots, IDX)], greedy=True)
        eng._candidates = spy
        eng.decode_pp(slots, [len(cases[i]["prompt"]) for i in IDX], torch.tensor([g[0] for g in gold], device=dev),
                      [len(gold[0]) - 1] * len(IDX), greedy=True, force=dict(zip(slots, gold)))
    if rank == 1:
        # Calls alternate group 0 (slots 0-1), group 1 (slots 2-3); call 2j + g is step j.
        errs = np.zeros((len(gold[0]) - 1, len(IDX)))
        for n, lp in enumerate(rows):
            j, g = divmod(n, 2)
            for r in range(lp.shape[0]):
                s = 2 * g + r
                errs[j, s] = abs(lp[r, gold[s][j + 1]].item() - cases[IDX[s]]["logprobs"][j + 1])
        print(f"split {split}: logprob err mean per case {errs.mean(0).round(3).tolist()} max {errs.max():.3f}", flush=True)
        print("FAIL" if errs.mean() > 0.15 or len(rows) != 2 * (len(gold[0]) - 1) else "PASS", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with open("tests/golden/golden.json") as f:
        cases = json.load(f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, cases, int(sys.argv[1]) if len(sys.argv) > 1 else 24), nprocs=2)
