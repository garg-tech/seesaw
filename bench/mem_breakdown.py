"""Where GPU memory goes in the PP layout, per rank.

  python bench/mem_breakdown.py <split> <slots> [lean]

Builds the engine the way the server does in seesaw mode (TP graphs, then PP) and
prints the size of each part and what is left free.
"""
import socket
import sys

import torch
import torch.multiprocessing as mp

sys.path.insert(0, ".")
GB = 2 ** 30


def size(x):
    if isinstance(x, torch.Tensor):
        return x.nbytes
    if isinstance(x, dict):
        return sum(size(v) for v in x.values())
    return 0


def worker(rank, port, split, slots, lean):
    from seesaw.engine import Engine, init_worker
    dev = init_worker(rank, port)
    eng = Engine(rank, dev, slots=slots, max_len=7200, chunk=2048, mode="tp", graphs=True, split=split,
                 lean=lean)
    eng.to_pp()
    torch.cuda.synchronize()
    r, kv, L = eng.resharder, eng.dec.kv, eng.dec.layers
    full = sum(size(kv.k[l.spec.idx]) + size(kv.v[l.spec.idx]) for l in L if l.spec.full)
    sliding = sum(size(kv.k[l.spec.idx]) + size(kv.v[l.spec.idx]) for l in L if not l.spec.full)
    weights = sum(size(l.h) for l in L)
    parts = {
        "embedding": eng.dec.embed_w.nbytes,
        "layer weights held now": weights,
        "full-layer KV": full,
        "sliding KV held now": sliding,
    }
    free, total = torch.cuda.mem_get_info()
    lines = [f"rank {rank}  split {split}  slots {slots}  lean {lean}"]
    lines += [f"  {k:34s} {v / GB:6.2f} GB" for k, v in parts.items()]
    lines += [f"  {'allocated':34s} {torch.cuda.memory_allocated() / GB:6.2f} GB",
              f"  {'reserved':34s} {torch.cuda.memory_reserved() / GB:6.2f} GB",
              f"  {'free on device':34s} {free / GB:6.2f} GB of {total / GB:.2f}",
              f"  per slot: full KV {full / slots / 2**20:.0f} MB, sliding {sliding / slots / 2**20:.0f} MB"
              f" (incl. scratch)"]
    for i in range(2):
        torch.distributed.barrier()
        if i == rank:
            print("\n".join(lines), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, int(sys.argv[1]), int(sys.argv[2]), "lean" in sys.argv[3:]), nprocs=2)
