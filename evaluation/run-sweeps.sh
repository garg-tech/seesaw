#!/usr/bin/env bash
# Run several engine configurations back to back: start the engine on the GPU machine,
# wait for it, run evaluation/vllm-bench.sh unchanged, stop it. Re-running skips
# levels that already have a complete result.
#
#   evaluation/run-sweeps.sh "seesaw_28L|seesaw --split 28|1 2 4 8 16 32 64 128" \
#                            "engine_tp_f|tp|1 2 4 8 16 32 64 128"
#   each argument: CONFIG name | scripts/serve-engine.sh arguments | concurrency levels
#
# With BOX_HOST set (.env) the engine is driven over SSH; empty, it runs on this machine.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh

stop_engine () {
  on_box 'tmux kill-session -t engine 2>/dev/null
    for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && exit 0; sleep 1; done
    echo "GPU memory still held:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 1'
}

for spec in "$@"; do
  IFS='|' read -r config args levels <<<"$spec"
  echo "##### $config: serve-engine.sh $args, levels $levels  ($(date -Is))"
  stop_engine || exit 1
  on_box "scripts/serve-engine.sh $args" | tail -1
  ready=0
  for _ in $(seq 1 90); do
    curl -sf -m 3 "$BASE_URL/v1/models" | grep -q gemma4-12b && { ready=1; break; }
    sleep 5
  done
  if [ $ready = 0 ]; then
    echo "ERROR: $config did not come up"; on_box 'tail -20 $(ls -t logs/engine_* | head -1)'
    continue
  fi
  # GPU clocks and power every 5 s, next to the results: the 70 W card throttles.
  mkdir -p "results/$config"
  on_box "nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 5" \
    > "results/$config/gpu_$(date +%Y%m%d_%H%M%S).csv" 2>/dev/null &
  smi=$!
  CONFIG="$config" LEVELS="$levels" evaluation/vllm-bench.sh
  echo "##### $config exit $?  ($(date -Is))"
  kill $smi 2>/dev/null
  on_box 'L=$(ls -t logs/engine_* | head -1); echo "engine log $L: tracebacks $(grep -c Traceback $L)"'
done
stop_engine
