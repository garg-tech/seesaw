#!/usr/bin/env bash
# Copy the engine, tests and scripts to the GPU machine (BOX_HOST:BOX_DIR from .env),
# for when you edit on a machine without GPUs. Logs and the virtualenv there are kept.
#
#   scripts/sync-box.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
[ -n "$BOX_HOST" ] || { echo "ERROR: set BOX_HOST in .env"; exit 1; }
rsync -az --delete --exclude __pycache__ seesaw tests bench scripts requirements "$BOX_HOST:$BOX_DIR/"
echo "synced to $BOX_HOST:$BOX_DIR"
