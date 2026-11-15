# Results

Raw output of every benchmark run quoted in the report. Each run is
`evaluation/vllm-bench.sh` (`vllm bench serve` 0.31.0, random dataset, input 6144 and
output 512 or 1024 tokens, `--ignore-eos`, 10 × concurrency prompts per level) against one
server configuration.

| Folder | Server | What it is | Dates |
|---|---|---|---|
| `tp2/` | vLLM 0.31.0, `--tensor-parallel-size 2` | **Baseline.** Best vLLM layout at every level | 2026-10-23 |
| `pp2/` | vLLM 0.31.0, `--pipeline-parallel-size 2` | **Baseline.** Also `in6k_out1` (prefill only) | 2026-10-22 to 10-30 |
| `pp2_28-20/`, `pp2_30-18/` | vLLM PP2 with an uneven layer split (`VLLM_PP_LAYER_PARTITION`) | Prefill only (`in6k_out1`): how much an uneven split speeds up prefill | 2026-10-30, 10-31 |
| `seesaw_28L/` | Seesaw engine, `--mode seesaw --split 28` (lean: KV parking, split embedding) | **Final Seesaw configuration** | 2026-11-04 to 11-05 |
| `seesaw_26f/` | Seesaw engine, `--mode seesaw --split 26`, fused | Seesaw before parking | 2026-11-03 |
| `engine_tp_f/` | Seesaw engine, `--mode tp`, fused | Control: same engine, TP only, never re-shards | 2026-11-03 to 11-04 |
| `engine_pp_f/` | Seesaw engine, `--mode pp --split 26`, fused | Control: same engine, PP only, never re-shards | 2026-11-03 to 11-04 |
| `seesaw/` | Seesaw engine, `--split 24`, before fusion | First full Seesaw sweep (24/24) | 2026-11-02 |
| `seesaw_26/` | Seesaw engine, `--split 26`, before fusion | First uneven split | 2026-11-02 to 11-03 |
| `engine_tp/` | Seesaw engine, `--mode tp`, before fusion | First control, c = 1, 4, 16, 32, 64 | 2026-11-02 |
| `compare_final/` | | Comparison plots of the six final configurations | |

In each folder:

- `<test>_n<c>.json`: the `vllm bench serve` result at concurrency `c`. `total_token_throughput`
  (input + output tokens per second) is the headline metric; TTFT, TPOT, ITL and end-to-end
  latency percentiles are in the same file.
- `server_info_<timestamp>.json`: what the client saw at the start of each run (server
  version, model, KV capacity). The server address is replaced by `gpu-box`.
- `gpu_<timestamp>.csv`: `nvidia-smi` every 5 s during the run (SM clock, power,
  utilisation, memory), for the runs made through `evaluation/run-sweeps.sh`.
- `plots/`: per-configuration plots and `<config>_<test>_summary.csv`.
- `*.failed-<timestamp>`: a result with failed requests, kept and not quoted; the level was
  rerun. `pp2/old_blocks12835/` holds PP2 levels from a server start that got a different KV
  capacity (12835 blocks), superseded by the rerun in `pp2/`.
- `seesaw_28L/no_gpu_log/`: the first 1024-out c = 64 and 128 results, run while SSH to the
  GPU machine was down, so without a GPU log. Rerun with the log; the two agree within 0.1%.

Every level in the six final configurations completed all its requests with 0 failures.

## Regenerate

```bash
python evaluation/make_tables.py                   # throughput tables (Markdown)
python evaluation/make_tables.py --metric ttft     # median TTFT
for t in in6k_out512 in6k_out1024; do
  .venv-bench/bin/python evaluation/plot_results.py --compare \
    results/{seesaw_28L,seesaw_26f,engine_tp_f,engine_pp_f,tp2,pp2}/plots/*_${t}_summary.csv \
    --baseline tp2 pp2 --tag $t --out-dir results/compare_final
done
```
