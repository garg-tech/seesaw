#!/usr/bin/env bash
# One-time setup of the benchmark client (the machine that runs vllm-bench.sh).
#
#   ./setup-bench-client.sh
#
# - creates .venv-bench with the same vllm version the server runs (so the
#   `vllm bench serve` client and the server agree on API details) + matplotlib
# - pre-fetches the Gemma 4 tokenizer files (not the 24 GB weights), so the
#   first sweep doesn't stall on a download
# - checks the server is reachable
#
# vllm installs CUDA wheels even on a GPU-less client; that's expected. The
# bench client never touches a GPU (`vllm --version` may fail to infer a device
# here, `vllm bench serve` does not need one).
set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="${BASE_URL:-http://gpu-box:8000}"
TOKENIZER="${TOKENIZER:-google/gemma-4-12B-it}"

SERVER_VER=$(curl -s -m 5 "$BASE_URL/version" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null || true)
VLLM_VER="${VLLM_VER:-${SERVER_VER:-0.31.0}}"
echo "server at $BASE_URL: ${SERVER_VER:-unreachable}; installing client vllm==$VLLM_VER"

command -v uv >/dev/null || { echo "needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv-bench ] || uv venv --python 3.11 .venv-bench
VIRTUAL_ENV=.venv-bench uv pip install "vllm==$VLLM_VER" matplotlib

.venv-bench/bin/python - "$TOKENIZER" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
repo = sys.argv[1]
snapshot_download(repo, allow_patterns=["tokenizer*", "*.jinja", "config.json",
                                        "generation_config.json", "special_tokens_map.json"])
tok = AutoTokenizer.from_pretrained(repo)
print(f"tokenizer ok: {repo}, vocab {len(tok)}")
EOF

echo "client ready. next: start the server on the GPU box (CONFIG=pp2 ./serve.sh), then CONFIG=pp2 ./vllm-bench.sh"
