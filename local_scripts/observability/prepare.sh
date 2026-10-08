#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-13.2
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p .xpool-cache
wrapper_log=$(mktemp .xpool-cache/observer-prepare.XXXXXX.log)
printf 'wrapper log: %s\n' "$wrapper_log"
prepare() {
    uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool \
        --config-settings-package xpool:cmake.define.XPOOL_CUDA_ARCHITECTURES=80-real || return "$?"
    uv run --no-sync python -u -m local_scripts.observability.remote prepare "$@"
}
prepare "$@" 2>&1 | tee "$wrapper_log"
