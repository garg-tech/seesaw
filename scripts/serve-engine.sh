#!/usr/bin/env bash
# Start the Seesaw engine on the GPU machine, in a tmux session called "engine".
#
#   scripts/serve-engine.sh seesaw --split 28     # the configuration in the report
#   scripts/serve-engine.sh tp                    # same engine, TP only (control)
#   scripts/serve-engine.sh pp --split 26         # same engine, PP only (control)
#
# Any arguments after the mode go to `python -m seesaw.server` (see --help there).
# Stop it with:  tmux kill-session -t engine
# Log: logs/engine_<mode>_<timestamp>.log
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-seesaw}"; shift || true
VENV="${SEESAW_VENV:-.venv}"
[ -f "$VENV/bin/activate" ] || { echo "ERROR: no virtualenv at $VENV (set SEESAW_VENV, see README)"; exit 1; }
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$BUSY" ]; then echo "ERROR: GPU memory is held by:"; echo "$BUSY"; exit 1; fi
if tmux has-session -t engine 2>/dev/null; then echo "ERROR: tmux session 'engine' already exists"; exit 1; fi
mkdir -p logs
LOG="logs/engine_${MODE}_$(date +%Y%m%d_%H%M%S).log"
# expandable_segments: without it 26/22 and 28/20 do not fit 32 sequences on GPU0.
tmux new-session -d -s engine "source $VENV/bin/activate; export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True; python -m seesaw.server --mode $MODE $* 2>&1 | grep --line-buffered -v Warning | tee $LOG"
echo "started in tmux session 'engine', log $LOG"
