#!/usr/bin/env python3
"""Check NCCL between the two GPUs, then measure what the link gives.

  python bench/nccl_check.py
  NCCL_DEBUG=INFO python bench/nccl_check.py 2>&1 | tee nccl.log   # when it fails
  NCCL_P2P_DISABLE=1 python bench/nccl_check.py                    # route via host memory

Step 1 is the same init + tiny all_reduce that vLLM does when it builds its
process groups, so if this passes, vLLM's NCCL setup will too. Step 2 times
the three patterns the plan cares about: all_reduce (TP), one-way send (PP
activations), and a two-way exchange (PP<->TP re-sharding).
"""
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [1, 16, 64, 256]


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=dev)

    x = torch.ones(1, device=dev)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert x.item() == 2, x.item()
    if rank == 0:
        print("step 1: NCCL init + all_reduce OK\n", flush=True)
        print(f"{'size':>7}  {'all_reduce':>18}  {'send 0->1':>10}  {'exchange (each way)':>20}")

    peer = 1 - rank
    for mb in SIZES_MB:
        t = torch.ones(mb * 2**20 // 2, dtype=torch.bfloat16, device=dev)
        buf = torch.empty_like(t)

        def send():
            if rank == 0:
                dist.send(t, peer)
            else:
                dist.recv(buf, peer)

        def exchange():
            ops = [dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, buf, peer)]
            for r in dist.batch_isend_irecv(ops):
                r.wait()

        ar, p2p, ex = timed(lambda: dist.all_reduce(t)), timed(send), timed(exchange)
        if rank == 0:
            gb = mb / 1024
            print(f"{mb:>4} MB  {ar * 1e3:7.2f} ms {gb / ar:5.1f} GB/s  "
                  f"{gb / p2p:5.1f} GB/s  {gb / ex:15.1f} GB/s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), "
          f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
    print(f"  peer access 0->1: {torch.cuda.can_device_access_peer(0, 1)}")
    print("  NCCL env:", {k: v for k, v in os.environ.items() if k.startswith("NCCL_")} or "none")
    print()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2)
