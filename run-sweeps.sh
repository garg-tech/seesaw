#!/usr/bin/env bash
# Run several engine configs back to back from the client: start the engine on the box,
# wait for it, run vllm-bench.sh unchanged, stop it. Re-running skips finished levels.
#
#   ./run-sweeps.sh "seesaw_f|seesaw --split 26|1 2 4 8 16 32 64 128" "engine_tp_f|tp|1 4 16 32 64"
#   each argument: CONFIG name | serve-engine.sh args | levels
set -uo pipefail
cd "$(dirname "$0")"
BASE_URL="${BASE_URL:-http://gpu-box:8000}"

stop_engine () {
  ssh gpu-box 'tmux kill-session -t engine 2>/dev/null
    for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && exit 0; sleep 1; done
    echo "GPU memory still held:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 1'
}

for spec in "$@"; do
  IFS='|' read -r config args levels <<<"$spec"
  echo "##### $config: serve-engine.sh $args, levels $levels  ($(date -Is))"
  stop_engine || exit 1
  ssh gpu-box "cd ~/devansh/seesaw-engine && ./serve-engine.sh $args" | tail -1
  ready=0
  for _ in $(seq 1 90); do
    curl -sf -m 3 "$BASE_URL/v1/models" | grep -q gemma4-12b && { ready=1; break; }
    sleep 5
  done
  if [ $ready = 0 ]; then
    echo "ERROR: $config did not come up"; ssh gpu-box 'tail -20 $(ls -t ~/devansh/seesaw-engine/logs/engine_* | head -1)'
    continue
  fi
  # GPU clocks and power every 5 s, next to the results (PLAN.md section 7).
  mkdir -p "results/$config"
  ssh gpu-box "nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 5" \
    > "results/$config/gpu_$(date +%Y%m%d_%H%M%S).csv" 2>/dev/null &
  smi=$!
  CONFIG="$config" LEVELS="$levels" ./vllm-bench.sh
  echo "##### $config exit $?  ($(date -Is))"
  kill $smi 2>/dev/null
  ssh gpu-box 'L=$(ls -t ~/devansh/seesaw-engine/logs/engine_* | head -1); echo "engine log $L: tracebacks $(grep -c Traceback $L)"'
done
stop_engine
