#!/usr/bin/env bash
# vllm bench serve — concurrency sweep against a remote server.
# Run this from a machine that is NOT the GPU box (or at least a separate process).
#
#   ./setup-bench-client.sh            # once: venv with the vllm CLI + tokenizer check
#   CONFIG=pp2 ./vllm-bench.sh         # after `CONFIG=pp2 ./serve.sh` on the GPU box
#   CONFIG=tp2 ./vllm-bench.sh
#
# Results land in results/<CONFIG>/<test>_n<N>.json, plots and a summary CSV in
# results/<CONFIG>/plots/. Re-running skips levels that already have a result.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:?set CONFIG to the server config being measured, e.g. CONFIG=pp2}"
BASE_URL="${BASE_URL:-http://gpu-box:8000}"
MODEL="${MODEL:-gemma4-12b}"                    # must match --served-model-name on the server
TOKENIZER="${TOKENIZER:-google/gemma-4-12B-it}" # HF id or local path; must match the served model
OUT_DIR="${OUT_DIR:-results/$CONFIG}"
LEVELS="${LEVELS:-1 2 4 8 16 32 64 128}"
TESTS="${TESTS:-in6k_out512 in6k_out1024}"     # also available: random sharegpt
PROMPTS_MULT="${PROMPTS_MULT:-10}"             # num-prompts = N * PROMPTS_MULT
# completions keeps the 6144-token input exact (no chat template) and is the
# endpoint the Seesaw engine will expose, so every system sees the same client.
API="${API:-completions}"                      # completions | chat

if [ -x .venv-bench/bin/vllm ]; then VLLM=.venv-bench/bin/vllm; PY=.venv-bench/bin/python
else VLLM=vllm; PY=python3; fi

case "$API" in
  completions) BACKEND=(--backend openai --endpoint /v1/completions) ;;
  chat)        BACKEND=(--backend openai-chat --endpoint /v1/chat/completions) ;;
  *) echo "API must be completions or chat"; exit 1 ;;
esac

# Fail fast if the server is down or serving a different model.
SERVED=$(curl -s -m 5 "$BASE_URL/v1/models" || true)
if ! grep -q "\"id\":\"$MODEL\"" <<<"$SERVED"; then
  echo "ERROR: $BASE_URL is not serving '$MODEL'. /v1/models said:"
  echo "  ${SERVED:-<no response>}"
  exit 1
fi

# /reset_prefix_cache is a dev-only route: it only exists if the server was
# started with VLLM_SERVER_DEV_MODE=1. Without it every run inherits the
# previous run's warm cache, so check up front rather than find out later.
# The route always answers 200; whether the reset happened is in the body
# ({"success": false} while requests still hold blocks).
reset_cache () {
  local body
  for _ in 1 2 3 4 5; do
    body=$(curl -s -m 10 -w "\n%{http_code}" -X POST "$BASE_URL/reset_prefix_cache" || echo 000)
    [ "$(tail -1 <<<"$body")" != "200" ] && { echo "$(tail -1 <<<"$body")"; return; }
    grep -q '"success":true' <<<"$body" && { echo ok; return; }
    sleep 2
  done
  echo "not-reset"
}
RESET=$(reset_cache)
if [ "$RESET" != "ok" ] && [ "$RESET" != "not-reset" ]; then
  echo "ERROR: POST /reset_prefix_cache returned $RESET."
  echo "  Restart the server with VLLM_SERVER_DEV_MODE=1 to enable it. Without it"
  echo "  the sweep runs with a warm prefix cache and the TTFT curve is wrong."
  exit 1
fi

# What was measured, next to the numbers.
mkdir -p "$OUT_DIR"
{
  echo "{\"config\": \"$CONFIG\", \"base_url\": \"$BASE_URL\", \"date\": \"$(date -Is)\","
  echo " \"server_version\": $(curl -s "$BASE_URL/version"), \"models\": $SERVED,"
  # KV capacity caps the high-concurrency levels, and it has varied between
  # restarts of the same config, so record it with every run.
  echo " \"num_gpu_blocks\": \"$(curl -s "$BASE_URL/metrics" | grep -o 'num_gpu_blocks="[0-9]*"' | head -1 | grep -o '[0-9]*')\","
  echo " \"client_vllm\": \"$($VLLM --version 2>/dev/null | tail -1)\", \"api\": \"$API\"}"
} > "$OUT_DIR/server_info_$(date +%Y%m%d_%H%M%S).json"

# A level counts as done only if every request succeeded. vllm bench serve
# still writes a result when the server dies mid-level, just with failures.
complete () {     # complete <result.json>
  [ -s "$1" ] && "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(d.get("failed", 0) != 0 or d["completed"] != d.get("num_prompts", d["completed"]))
' "$1"
}

# When the server dies mid-level, the client's open streams never close and
# vllm bench serve waits on them for hours. Kill it once /health stops answering.
watchdog () {     # watchdog <bench pid>
  local misses=0
  while kill -0 "$1" 2>/dev/null; do
    sleep 30
    if curl -sf -m 10 -o /dev/null "$BASE_URL/health"; then misses=0; else misses=$((misses + 1)); fi
    if [ "$misses" -ge 3 ]; then
      echo "ERROR: server stopped answering $BASE_URL/health; aborting this level"
      kill "$1" 2>/dev/null
      return
    fi
  done
}

run () {          # run <tag> <extra args...>
  local tag="$1"; shift
  for N in $LEVELS; do
    local f="$OUT_DIR/${tag}_n${N}.json"
    if complete "$f"; then
      echo "=== $CONFIG/$tag  concurrency=$N  -- already done, skipping ==="
      continue
    fi
    if [ -e "$f" ]; then
      mv "$f" "$f.failed-$(date +%Y%m%d_%H%M%S)"
      echo "=== $CONFIG/$tag  concurrency=$N  -- previous result had failed requests, set aside ==="
    fi
    echo "=== $CONFIG/$tag  concurrency=$N  ($(( N * PROMPTS_MULT )) prompts) ==="
    [ "$(reset_cache)" = ok ] || echo "WARNING: prefix cache reset did not succeed"
    "$VLLM" bench serve \
      "${BACKEND[@]}" \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --max-concurrency "$N" \
      --num-prompts $(( N * PROMPTS_MULT )) \
      --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,95,99 \
      --save-result \
      --result-dir "$OUT_DIR" \
      --result-filename "${tag}_n${N}.json" \
      "$@" &
    local bench=$!
    watchdog "$bench" &
    local wd=$!
    wait "$bench" || true
    kill "$wd" 2>/dev/null || true
    complete "$f" || echo "WARNING: $f has failed requests; rerun this script to redo the level"
    if ! curl -sf -m 10 -o /dev/null "$BASE_URL/health"; then
      echo "ERROR: the server is down. Restart it, then rerun this script to resume from this level."
      exit 1
    fi
    sleep 5
  done
  "$PY" plot_results.py "$OUT_DIR/${tag}_n*.json" \
    --tag "${CONFIG}_${tag}" --label "$CONFIG" --out-dir "$OUT_DIR/plots"
}

# ---------------------------------------------------------------- warmup
reset_cache > /dev/null
"$VLLM" bench serve "${BACKEND[@]}" --base-url "$BASE_URL" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 20 --max-concurrency 4 --ignore-eos

for T in $TESTS; do
  case "$T" in
    # ------------------------------------------------------------ Seesaw workload
    # Fixed lengths (range ratio 0) and no shared prefix: every config does the
    # same prefill and decode work, which is what the comparison needs.
    # Any in<len>_out<len>, "k" = x1024: in6k_out512, in6k_out1 (prefill only),
    # in128_out1024 (decode heavy).
    in*_out*)
      IN=${T#in}; IN=${IN%%_*}; OUT=${T##*_out}
      case "$IN" in *k) IN=$(( ${IN%k} * 1024 )) ;; esac
      run "$T" \
        --dataset-name random \
        --random-input-len "$IN" \
        --random-output-len "$OUT" \
        --random-range-ratio 0
      ;;
    # ------------------------------------------------------------ coding-ish shape
    # --random-prefix-len gives every request a shared prefix,
    # without it the prefix cache is on but never hits.
    random)
      run random \
        --dataset-name random \
        --random-input-len 4096 \
        --random-output-len 512 \
        --random-prefix-len 2048
      ;;
    # ------------------------------------------------------------ ShareGPT
    # One-time download:
    #   wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    sharegpt)
      run sharegpt \
        --dataset-name sharegpt \
        --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 512
      ;;
    *) echo "unknown test '$T'"; exit 1 ;;
  esac
done

echo "done — JSON results in $OUT_DIR/, plots in $OUT_DIR/plots/"
echo "compare configs:  $PY plot_results.py --compare \"results/*/plots/*_in6k_out512_summary.csv\" --baseline tp2 pp2 --tag in6k_out512 --out-dir results/compare"
