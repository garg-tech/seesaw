# Architecture Decision Records

Project 24, group 25, CS5903 Distributed AI Training: a reimplementation of Seesaw
(dynamic model re-sharding, [arXiv 2503.06433](https://arxiv.org/abs/2503.06433)) for
Gemma 4 12B on two heterogeneous GPUs.

One record per material decision, in the order they were taken. Superseded records are
kept. All records were written down on 2026-11-15 from the project's working notes and
commit history; the decision dates are when each choice was made. Commit hashes refer to
this repository.

**Methodology record (9 October 2026 checkpoint).** Not applicable: work began after mid
presentation on 9th Oct.

| ID | Decision | Status |
|---|---|---|
| [ADR-001](#adr-001--evaluation-methodology-fixed-length-sweep-with-vllm-bench-serve) | Evaluation methodology: fixed-length sweep with `vllm bench serve` | Accepted |
| [ADR-002](#adr-002--own-engine-reusing-vllms-triton-kernels) | Own engine, reusing vLLM's Triton kernels | Accepted |
| [ADR-003](#adr-003--tp-half-tensors-as-the-storage-unit-in-both-layouts) | TP half-tensors as the storage unit in both layouts | Accepted |
| [ADR-004](#adr-004--re-shard-gpu-to-gpu-over-nccl-not-through-host-memory) | Re-shard GPU to GPU over NCCL, not through host memory | Accepted |
| [ADR-005](#adr-005--no-cpu-kv-buffer-prefill-a-batch-then-decode-it-to-completion) | No CPU KV buffer: prefill a batch, then decode it to completion | Accepted |
| [ADR-006](#adr-006--stay-in-tp-below-four-requests) | Stay in TP below four requests | Accepted |
| [ADR-007](#adr-007--uneven-pp-split-2622) | Uneven PP split 26/22 | Superseded by ADR-010 |
| [ADR-008](#adr-008--match-vllms-per-step-cost-cuda-graphs-and-torchcompile-fusion) | Match vLLM's per-step cost: CUDA graphs and torch.compile fusion | Accepted |
| [ADR-009](#adr-009--own-tp-and-pp-modes-as-same-engine-controls) | Own TP and PP modes as same-engine controls | Accepted |
| [ADR-010](#adr-010--2820-split-with-peer-kv-parking-and-a-split-embedding) | 28/20 split with peer KV parking and a split embedding | Accepted |

---

## ADR-001 — Evaluation methodology: fixed-length sweep with `vllm bench serve`

**Date, status, scope.** 2026-10-22, Accepted. Evaluation of every configuration
(`evaluation/`, `results/`).

**Context and drivers.** The project's claim is a throughput gain over vLLM, so the
measurement has to be identical for both systems, repeatable, and free of effects that
favour one layout by accident (prefix caching, variable lengths that change the work done).
The paper targets throughput-oriented, prefill-heavy workloads.

**Options considered.**
1. `vllm bench serve` against the OpenAI-compatible endpoint, random tokens, fixed lengths
   (input 6144, output 512 and 1024), `--ignore-eos`, concurrency 1 to 128.
2. ShareGPT conversations (variable lengths), as in many serving papers.
3. Offline batch throughput (submit everything at once), as in the paper's evaluation.

**Decision and rationale.** Option 1. Every system receives the same requests from the same
client, and every request costs the same work, so differences come from the server alone.
Concurrency 1 to 128 shows both the latency-bound and the throughput-bound regime; option 3
would only show the latter. Rules: 10 × c prompts per level, prefix cache off in vLLM (the
engine has none), and a level counts only with 0 failed requests. Headline metric: total
(input + output) tokens per second; TTFT and TPOT are reported too. GPU clocks and power are
logged because GPU1 is capped at 70 W.

**Consequences and trade-offs.** Fixed output lengths suit our round-based scheduler
(ADR-005): a whole batch finishes on the same step. With variable lengths it would leave
slots idle, so these results do not show that weakness; the report says so. The paper's
offline datasets are not reproduced.

**Evidence.** `evaluation/vllm-bench.sh`; raw results in `results/`; `evaluation/make_tables.py`.

**Revisit trigger.** If the workload of interest has variable output lengths, add a
ShareGPT-style test (`TESTS=sharegpt` already exists in `vllm-bench.sh`) before quoting gains.

---

## ADR-002 — Own engine, reusing vLLM's Triton kernels

**Date, status, scope.** 2026-11-01, Accepted. Whole engine (`seesaw/`).

**Context and drivers.** Re-sharding needs full control over where every weight and KV
tensor lives, and over when a batch switches layout. vLLM's executor, KV manager and
scheduler assume one fixed parallel layout for the life of the server. Constraint: about
one week to build, and the engine must be within about 10% of vLLM's per-step speed, or
the modelled gain disappears.

**Options considered.**
1. Modify vLLM to switch between PP and TP at runtime.
2. Own engine in plain PyTorch (SDPA attention).
3. Own engine for model, KV layout, re-shard, scheduler and server, calling vLLM's Triton
   attention kernels as a library.

**Decision and rationale.** Option 3. vLLM's Triton kernels already handle Gemma 4's
head_dim 512 full-attention layers, GQA and the 1024-token sliding window on both GPU
architectures (FlashAttention does not support head_dim 512 on SM 8.6 / 12.0). Option 1
touches too many subsystems for the time available (judgement). Option 2 would be slower at
long context (judgement, not measured).

**Consequences and trade-offs.** We own correctness, so the model is checked against vLLM's
greedy outputs (`tests/test_tp_golden.py`) and transformers (`tests/test_layers_vs_hf.py`).
The engine supports only Gemma 4 12B, text only, greedy and simple sampling, and no
continuous batching. It depends on vLLM 0.31's internal kernel APIs, which can change.

**Evidence.** Commits `2d5525e`, `d9e4873`. Decode step at B=1: 33.8 ms vs vLLM's 34 ms; our
TP mode reaches 0.99 to 1.07 of vLLM TP2's throughput (`results/engine_tp_f`).

**Revisit trigger.** vLLM gains runtime re-sharding, or a vLLM upgrade breaks the kernel
calls in `seesaw/model/layers.py`.

---

## ADR-003 — TP half-tensors as the storage unit in both layouts

**Date, status, scope.** 2026-11-01, Accepted. `seesaw/model/`, `seesaw/kv_cache.py`,
`seesaw/reshard.py`.

**Context and drivers.** Every switch moves weights and KV between the two GPUs. The move
must be fast, allocate no memory, and leave both layouts numerically identical.

**Options considered.**
1. Store each layer whole (PP layout) and slice into TP shards at each switch.
2. Store two layouts' worth of tensors and copy between them.
3. Store every layer's weights and KV as two TP halves, `h0` and `h1`, in both modes. A TP
   GPU holds one half of every layer; a PP GPU holds both halves of its layers and adds
   the two partial results itself.

**Decision and rationale.** Option 3. There is one forward path; PP and TP differ only in
where the sum of the halves happens (locally or by all-reduce). Re-sharding swaps whole
tensors, with no slicing or repacking, and an incoming half lands in the slot an outgoing
one frees. Gemma 4's full-attention layers have one KV head, which cannot be split, so their
K/V projection and cache are replicated and only Q and the output projection are halved.

**Consequences and trade-offs.** PP does two half-GEMMs per projection instead of one
(fusion in ADR-008 recovers most of the cost). Full-layer KV exists on both GPUs during TP.

**Evidence.** Commit `0c8bbc1`; `tests/test_reshard.py` (prefill PP, switch, decode TP,
switch back, twice, against vLLM's outputs). Switch time 1.25 s with 3 sequences, 2.0 s
with 32.

**Revisit trigger.** A model whose layers cannot be split in two this way, or more than two
GPUs.

---

## ADR-004 — Re-shard GPU to GPU over NCCL, not through host memory

**Date, status, scope.** 2026-11-01, Accepted. `seesaw/reshard.py`.

**Context and drivers.** The paper reloads weight shards from a copy in CPU memory and moves
KV through CPU shared memory. Our machine has 15 GB of host RAM and the model is 23.9 GB in
bf16, so a full host copy does not fit. The GPUs sit on different sockets with no peer
access; NCCL goes through `/dev/shm` (measured 8.4 GB/s each way for a two-way exchange).

**Options considered.**
1. The paper's method: keep weights in host RAM, reload the needed shards.
2. Reload shards from disk on every switch.
3. Exchange the halves that change owner directly between the GPUs with NCCL
   send/receive, on the same communicator used for TP all-reduce and PP activations.

**Decision and rationale.** Option 3. Option 1 does not fit in memory. Option 2 would read
several GB from disk per switch (judgement: slower than the measured NCCL exchange). The
exchange moves about 6.3 GB from GPU0 and 4.5 GB from GPU1 at 28/20, in pieces of at most
64 MB so the staging buffers stay small.

**Consequences and trade-offs.** Both GPUs stop computing during a switch (1.8 to 2.0 s
at 32 sequences, about 2% of a round). The switch cannot be overlapped with compute the way
the paper's asynchronous CPU pipeline can.

**Evidence.** `bench/nccl_check.py` (link speeds); switch times in the engine log
(`round ... switch 1.8s`); `tests/test_reshard.py`.

**Revisit trigger.** At least about 64 GB of host RAM (then the paper's design, ADR-005
option 1, becomes possible), or GPUs with peer-to-peer access.

---

## ADR-005 — No CPU KV buffer: prefill a batch, then decode it to completion

**Date, status, scope.** 2026-11-01, Accepted. `seesaw/scheduler.py`.

**Context and drivers.** The paper's tiered KV cache buffering stores prefilled KV in CPU
memory so a large batch can be prefilled in PP before one switch to TP, and decode refills
from the buffer as sequences finish. It provisions 80 GiB of CPU memory per GPU for this.
We have 15 GB in total, about 11 GB free: room for 10 to 15 sequences of 6144 tokens.

**Options considered.**
1. The paper's CPU KV buffer, scaled to available RAM.
2. Continuous batching with prefill-first scheduling, switching layout whenever the phase
   changes (the paper's Figure 2a: too many switches).
3. Rounds: admit up to the free GPU slots, prefill them in PP, switch once, decode the whole
   batch to completion in TP, then start the next round (the paper's "decode-prioritizing"
   scheduling, Figure 2b).

**Decision and rationale.** Option 3. Option 1 would add at most 10 to 15 sequences per
round; with a switch costing about 2% of a round, the gain is at most about 2% for this
workload (estimate, not built). Option 2 would switch far too often. The GPU slots act as
the buffer: PP to TP happens when the slots are full or the queue is empty; TP to PP when
the decode batch has finished.

**Consequences and trade-offs.** One switch serves at most 32 sequences, far fewer than the
paper's buffer allows. Requests that arrive during decode wait for the next round, so median
TTFT is worse than vLLM's at c ≤ 32 (13.4 s vs 5.0 s at c = 16). With variable output
lengths, finished sequences would leave slots idle until the round ends (ADR-001).

**Evidence.** `tests/test_scheduler.py`; round timings in the engine log (`round B=32
layout=pp prefill 46.2s switch 1.8s decode 60.0s` at 28/20).

**Revisit trigger.** More host RAM (≥ 64 GB), or a workload with variable output lengths;
either makes refilling decode from a buffer worthwhile.

---

## ADR-006 — Stay in TP below four requests

**Date, status, scope.** 2026-11-02, Accepted. `--min-switch-batch` in `seesaw/server.py`.

**Context and drivers.** A switch to PP and back costs 1.25 to 2.0 s. PP prefill saves about
0.6 s per 6144-token request over TP prefill. With few requests the switch costs more than it
saves.

**Options considered.** 1. Always switch. 2. A fixed threshold. 3. A cost model evaluated per
round from measured constants.

**Decision and rationale.** Option 2 with threshold 4, from the break-even
`B × (prefill time TP − prefill time PP) > switch time`, which gives B ≈ 2 to 3; 4 adds a
margin. This is our rule; the paper does not define one. A per-round cost model (option 3)
would give the same answer for fixed-length requests.

**Consequences and trade-offs.** At c = 1 and 2 Seesaw is exactly our TP mode (333 vs 333
tok/s at c = 1, 512-out). For prompts much shorter than 6144 tokens the right threshold is
higher; it is a server flag.

**Evidence.** `tests/test_scheduler.py` (1 and 3 requests stay in TP, 8 switch);
`results/seesaw_28L` vs `results/engine_tp_f` at c = 1, 2.

**Revisit trigger.** Different prompt lengths or switch cost; then measure and reset the
threshold, or implement option 3.

---

## ADR-007 — Uneven PP split 26/22

**Date, status, scope.** 2026-11-03, **Superseded by ADR-010**. `--split` in the engine.

**Context and drivers.** GPU1 (RTX PRO 4000) is capped at 70 W and runs at about 1000 to
1500 MHz under sustained load, so in an even 24/24 PP split it is the slow stage. Moving
layers to GPU0 speeds up prefill but adds their weights and KV to GPU0, which is already the
memory limit at 32 slots.

**Options considered.** 24/24 (32 slots), 26/22, 27/21, 28/20 (24 slots).

**Decision and rationale.** 26/22 at 32 slots, with `expandable_segments` so it fits. PP
prefill rose from 3570 to about 3965 tok/s. 28/20 was faster (4347 tok/s) but fit only 24
sequences, and the model predicted 26/22 at 32 slots would win (2520 vs 2417 tok/s, 512-out).

**Consequences.** The sweep reached 1.17× vLLM TP2 at c ≥ 32 (24/24: 1.10×). Superseded when
parking made 28/20 fit 32 sequences (ADR-010).

**Evidence.** Commit `53c91d6`; `results/seesaw_26`, `results/seesaw` (24/24).

**Revisit trigger.** Replaced by ADR-010.

---

## ADR-008 — Match vLLM's per-step cost: CUDA graphs and torch.compile fusion

**Date, status, scope.** 2026-11-02 to 11-03, Accepted. `seesaw/model/`, `seesaw/decode_graph.py`.

**Context and drivers.** The first engine took 63 ms per decode step at B=1 against vLLM's
34 ms: about 90 small operations per layer were dispatched one by one from Python. Any gap
here shows up as a loss to vLLM that has nothing to do with re-sharding.

**Options considered.**
1. Accept the gap and compare only against our own TP mode.
2. Write custom fused CUDA kernels.
3. Do what vLLM does: CUDA graphs for decode, NCCL called directly on the compute stream,
   merged QKV and gate/up projections, and torch.compile on the small operations between
   GEMMs.

**Decision and rationale.** Option 3: no new kernels, and the same technique as the baseline.
Weights are merged on the CPU while loading, because concatenating on the GPU fragmented
memory and 26/22 stopped fitting.

**Consequences and trade-offs.** Startup takes longer (compilation and graph capture: about
two minutes until the server answers). CUDA graphs fix the batch sizes they were captured for.

**Evidence.** Commits `519e6c7`, `9e03607`. Decode step B=1: 63 → 38.2 → 33.8 ms (vLLM 34); B=32:
58.0 ms; TP prefill 2900 tok/s (vLLM 2870). Layer error vs transformers 0.84%, under the
0.93 to 1.01% bf16 rounding floor.

**Revisit trigger.** A vLLM or PyTorch upgrade that changes compile behaviour; rerun
`bench/engine_micro.py` against the numbers above.

---

## ADR-009 — Own TP and PP modes as same-engine controls

**Date, status, scope.** 2026-11-02 (TP), 2026-11-03 (PP), Accepted. `--mode tp|pp`.

**Context and drivers.** A gain over vLLM could come from re-sharding or from unrelated
differences between our engine and vLLM. The two must be separated.

**Options considered.** 1. Compare with vLLM only. 2. Also run the same engine fixed in TP and
fixed in PP, with everything else equal.

**Decision and rationale.** Option 2. The PP mode does what vLLM's PP does (two
micro-batches in flight, per-stage CUDA graphs, non-blocking sends), so it is a fair PP
baseline, not a strawman.

**Consequences and trade-offs.** Two more full sweeps (several hours of GPU time each). The own PP
mode uses rounds, not continuous batching, which likely explains its 0.80× of vLLM PP2 at
c = 2 (not measured).

**Evidence.** Commit `74d38f9`; `tests/test_pp_decode.py`; `results/engine_tp_f`,
`results/engine_pp_f`. At c ≥ 32, re-sharding is worth 1.16 to 1.22× over our TP and 1.15
to 1.24× over our PP (26/22 build), consistent with the gain over vLLM.

**Revisit trigger.** Engine changes that affect one mode only; rerun the controls.

---

## ADR-010 — 28/20 split with peer KV parking and a split embedding

**Date, status, scope.** 2026-11-04, Accepted. Supersedes ADR-007. `seesaw/reshard.py`,
`seesaw/kv_cache.py`, `seesaw/model/gemma4.py`, `seesaw/engine.py`.

**Context and drivers.** 28/20 prefills 10% faster than 26/22 but did not fit 32 sequences
on GPU0. In PP, GPU0's 28 layers pair with GPU1's 20 for the swap; the 8 unpaired "extra"
layers keep a spare copy of their second half (weights and KV) on GPU0. Measured with
`bench/mem_breakdown.py`: compared with 26/22 (4 extras), 28/20 adds about 0.83 GB of
weights on GPU0, and the extras' spare KV for 32 sequences takes about 1 GB. The paper's
answer, spilling KV to CPU, is not available (ADR-005).

**Options considered.**
1. Keep 26/22 (ADR-007).
2. 28/20 at 24 slots.
3. Park: the extras' `h1` KV is written to a small staging pool on GPU0 (about 233 blocks
   instead of 2338) and sent to GPU1, where TP needs it anyway, as soon as each prefill
   finishes. Parking alone gives about 30 slots.
4. Parking plus a split embedding: GPU0 keeps half of the 262144-row embedding (0.94 GB
   instead of 1.88 GB). TP looks tokens up vocab-parallel with one all-reduce, which is
   exact; in PP, GPU0 reads the other rows from host RAM.

**Decision and rationale.** Option 4, the only one that fits 32 sequences at 28/20 (0.79 GB
free on GPU0 after the switch). The other GPU's spare memory takes the role of the
paper's CPU tier, and parked KV no longer moves during the switch. The mode is called *lean*
and is used only in `--mode seesaw`, because GPU0 can no longer decode in PP.

**Consequences and trade-offs.** One extra all-reduce per decode step for the embedding
(no visible cost: c = 1 and 2 equal 26/22). More state to keep consistent in the KV tables;
covered by `tests/test_reshard.py 28 lean`.

**Evidence.** Commit `e2b1d0b`; `results/seesaw_28L`. PP prefill 4523 tok/s at B = 32 (26/22
lean: 4114). Full sweep, 0 failures: 1.31× vLLM TP2 at c ≥ 32 (512-out), 1.27 to 1.28×
(1024-out); 3 to 5% over 26/22.

**Revisit trigger.** A different GPU pair or model size, where GPU0 is no longer the memory
limit; rerun `bench/mem_breakdown.py` for the candidate splits.
