#!/usr/bin/env bash
# One-time setup of the benchmark client (the machine that runs vllm-bench.sh).
#
#   evaluation/setup-bench-client.sh
#
# - creates .venv-bench from requirements/client.txt: the vllm version the server
#   runs (so the `vllm bench serve` client and the server agree on API details),
#   plus matplotlib
# - pre-fetches the Gemma 4 tokenizer files (not the 24 GB weights), so the
#   first sweep doesn't stall on a download
# - checks the server is reachable
#
# vllm installs CUDA wheels even on a GPU-less client; that's expected. The
# bench client never touches a GPU (`vllm --version` may fail to infer a device
# here, `vllm bench serve` does not need one).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh

TOKENIZER="${TOKENIZER:-google/gemma-4-12B-it}"

command -v uv >/dev/null || { echo "needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv-bench ] || uv venv --python 3.11 .venv-bench
uv pip install --python .venv-bench -r requirements/client.txt

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

SERVER=$(curl -s -m 5 "$BASE_URL/version" || true)
echo "server at $BASE_URL: ${SERVER:-not reachable yet}"
echo "client ready. next: start a server on the GPU machine, then CONFIG=<name> evaluation/vllm-bench.sh"
