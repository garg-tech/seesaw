"""TP2 or PP2 across two workers must reproduce vLLM's greedy outputs.

  python tests/test_tp_golden.py [tp|pp]   # needs tests/golden/golden.json from bench/capture_golden.py

Teacher-forced on vLLM's tokens and compares log-probabilities, because near-ties
flip under bf16 rounding and one flip would make every later token differ.
"""
import json
import socket
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")


def worker(rank, port, cases, mode):
    from seesaw.engine import Engine, init_worker
    from seesaw.kv_cache import Batch
    dev = init_worker(rank, port)
    max_len = max(len(c["prompt"]) + len(c["tokens"]) for c in cases) + 16
    eng = Engine(rank, dev, slots=2, max_len=max_len, chunk=2048, mode=mode)
    report = 1 if mode == "pp" else 0     # the rank that sees logits
    orig = eng.sample
    cap = {}
    def spy(x, greedy, n):
        if x is not None:
            cap["lp"] = eng.logprobs(x)
        return orig(x, greedy, n)
    eng.sample = spy
    bad = 0
    with torch.inference_mode():
        for i, c in enumerate(cases):
            # Teacher-forced: feed vLLM's tokens so one near-tie can't derail the rest.
            eng.prefill([(0, c["prompt"])], greedy=True)
            n, errs, flips = len(c["prompt"]), [], 0
            gold = c["tokens"]
            for j, g in enumerate(gold):
                if rank == report:
                    lp = cap["lp"][0]
                    errs.append(abs(lp[g].item() - c["logprobs"][j]))
                    top = lp.argmax().item()
                    if top != g:
                        flips += 1
                        if flips <= 2:
                            print(f"   step {j}: ours {top} ({lp[top].item():.3f}) vs gold {g} (ours {lp[g].item():.3f}, vLLM {c['logprobs'][j]:.3f})")
                if j + 1 < len(gold):
                    eng.decode_step([0], [n], torch.tensor([g], device=dev), greedy=True)
                    n += 1
            eng.release([0])
            if rank == report:
                e = np.array(errs)
                print(f"case {i}: prompt {len(c['prompt'])}: logprob err mean {e.mean():.3f} max {e.max():.3f}, argmax flips {flips}/{len(gold)}", flush=True)
                bad += e.mean() > 0.15
    if rank == report:
        print("FAIL" if bad else "PASS", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with open("tests/golden/golden.json") as f:
        cases = json.load(f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, cases, sys.argv[1] if len(sys.argv) > 1 else "tp"), nprocs=2)
