#!/usr/bin/env bash
# Launch a vLLM baseline for the Seesaw comparison. Run this ON THE GPU BOX.
#
#   CONFIG=pp2 ./serve.sh          # pipeline parallel: layers split across both GPUs
#   CONFIG=tp2 ./serve.sh          # tensor parallel: every layer split across both GPUs
#   CONFIG=pp2 ./serve.sh --max-num-batched-tokens 4096   # extra args go to vllm serve
#
# Then, from the client machine:  CONFIG=pp2 ./vllm-bench.sh
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-pp2}"
MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
SERVED_NAME="${SERVED_NAME:-gemma4-12b}"      # what vllm-bench.sh passes as --model
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"        # 6144 in + 1024 out, with slack
GPU_UTIL="${GPU_UTIL:-0.90}"
# Prefix caching off by default: the Seesaw engine has none, and the random
# workload has no shared prefixes, so it would only add hashing work.
PREFIX_CACHE="${PREFIX_CACHE:-off}"           # off | on
LOG_DIR="${LOG_DIR:-logs}"

case "$CONFIG" in
  pp2) PAR=(--pipeline-parallel-size 2) ;;
  tp2) PAR=(--tensor-parallel-size 2) ;;
  *) echo "CONFIG must be pp2 or tp2"; exit 1 ;;
esac
case "$PREFIX_CACHE" in
  off) PC=(--no-enable-prefix-caching) ;;
  on)  PC=(--enable-prefix-caching) ;;
  *) echo "PREFIX_CACHE must be on or off"; exit 1 ;;
esac

# Use ./.venv if vllm isn't already on PATH.
if ! command -v vllm >/dev/null; then
  if [ -x .venv/bin/vllm ]; then
    source .venv/bin/activate
  else
    echo "ERROR: vllm not found on PATH or in $(pwd)/.venv. Install it first:"
    echo "  uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install vllm==0.31.0"
    exit 1
  fi
fi

if curl -s -m 2 -o /dev/null "http://localhost:$PORT/health"; then
  echo "ERROR: something is already serving on port $PORT:"
  curl -s -m 2 "http://localhost:$PORT/v1/models"; echo
  echo "Stop it first (both GPUs are needed)."
  exit 1
fi

# Workers of a crashed or Ctrl+C'd vllm run can outlive it and keep the weights
# resident; vllm then fails much later with a misleading free-memory error.
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
if [ -n "$BUSY" ]; then
  echo "ERROR: GPU memory is still held by:"
  while IFS=, read -r pid mem; do
    echo "  $mem  $(ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-120)"
  done <<<"$BUSY"
  echo "Kill those first (kill <pid>; kill -9 if they linger), then rerun."
  exit 1
fi

# Number GPUs the way nvidia-smi does: 0 = A5000 (17:00.0), 1 = PRO 4000 (BD:00.0).
# With PP2, stage 0 (first half of the layers) goes on GPU 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Enables POST /reset_prefix_cache, which vllm-bench.sh calls between levels.
export VLLM_SERVER_DEV_MODE=1
# The GPUs sit on different sockets (no peer access), so NCCL goes through
# /dev/shm. NCCL 2.29's cuMem host buffers fail to attach there ("attaching to
# shared memory segment /dev/shm/nccl- (size 0)"); the classic path works.
# Reproduce with: python bench/nccl_check.py
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
# Attention runs on Triton (FlashAttention can't do Gemma 4's head_dim 512 on
# SM 8.6/12.0), so FlashInfer is only used for the top-k/top-p sampler, and it
# has to JIT-compile that on first start. Neither the system CUDA (< 12.9, too
# old for SM 12.0) nor the venv's pip CUDA 13 (nvcc/nvvm 13.4 vs headers 13.0)
# can build it, so use vLLM's own sampler. Keep it the same for every baseline.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/serve_${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
{
  echo "# CONFIG=$CONFIG MODEL_ID=$MODEL_ID MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_UTIL=$GPU_UTIL PREFIX_CACHE=$PREFIX_CACHE extra: $*"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,clocks.max.sm --format=csv
  echo "vllm $(python3 -c 'import vllm; print(vllm.__version__)')"
  echo "env: $(env | grep -E '^(NCCL_|CUDA_MPS|VLLM_USE_|VLLM_PP_)' | tr '\n' ' ')"
} | tee "$LOG"

vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  "${PAR[@]}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --language-model-only \
  "${PC[@]}" \
  "$@" 2>&1 | tee -a "$LOG"
