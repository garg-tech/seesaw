#!/usr/bin/env bash
# Start the Seesaw engine on the GPU box, in tmux. Run it on the box, or from
# the client:  ssh gpu-box devansh/seesaw-engine/serve-engine.sh seesaw
#   serve-engine.sh <seesaw|tp> [extra server args]
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-seesaw}"; shift || true
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$BUSY" ]; then echo "ERROR: GPU memory is held by:"; echo "$BUSY"; exit 1; fi
mkdir -p logs
LOG="logs/engine_${MODE}_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s engine "source ../.venv/bin/activate; export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=false; python -m seesaw.server --mode $MODE $* 2>&1 | grep --line-buffered -v Warning | tee $LOG"
echo "started in tmux session 'engine', log $LOG"
