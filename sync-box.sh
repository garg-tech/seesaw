#!/usr/bin/env bash
# Copy the engine to the GPU box. Everything runs there; this machine has no GPUs.
#   ./sync-box.sh
set -euo pipefail
cd "$(dirname "$0")"
rsync -az --delete --exclude __pycache__ --exclude logs/ seesaw tests bench serve.sh serve-engine.sh gpu-box:devansh/seesaw-engine/
