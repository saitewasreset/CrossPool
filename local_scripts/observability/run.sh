#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p .xpool-cache
wrapper_log=$(mktemp .xpool-cache/observer-run.XXXXXX.log)
printf 'wrapper log: %s\n' "$wrapper_log"
uv run --no-sync python -u -m local_scripts.observability.remote run "$@" 2>&1 | tee "$wrapper_log"
