# Sourced by the scripts in scripts/ and evaluation/. Loads .env from the repository
# root, if there is one, without overriding variables already set in the environment,
# so `BASE_URL=... evaluation/vllm-bench.sh` still wins over .env.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ''|\#*) continue ;; esac
    value="${value%\"}"; value="${value#\"}"
    [ -z "${!key+x}" ] && [ -n "$value" ] && export "$key=$value"
  done < "$ROOT/.env"
fi
BASE_URL="${BASE_URL:-http://localhost:8000}"
BOX_HOST="${BOX_HOST:-}"
BOX_DIR="${BOX_DIR:-seesaw}"

# Run a shell command in the repository on the GPU machine: over SSH when BOX_HOST is
# set, otherwise right here (client and GPUs on the same machine).
on_box () {
  if [ -n "$BOX_HOST" ]; then ssh "$BOX_HOST" "cd $BOX_DIR && $1"
  else (cd "$ROOT" && bash -c "$1"); fi
}
