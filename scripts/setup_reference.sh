#!/usr/bin/env bash
set -euo pipefail
# Install uv 0.9.24 separately from https://github.com/astral-sh/uv/releases/tag/0.9.24.
# Set CUDA_HOME to your local CUDA 12.4.131 toolkit; keep private paths untracked.
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.4.131 toolkit directory}"
if [[ "$CUDA_HOME" == "xxxxxx" ]]; then
    echo "Replace the CUDA_HOME placeholder with your local CUDA 12.4.131 toolkit directory." >&2
    exit 1
fi
uv --version | rg '^uv 0\.9\.24$'
"$CUDA_HOME/bin/nvcc" --version | rg 'V12.4.131'
uv sync --frozen --extra eval
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-2}" \
    uv sync --frozen --extra eval --extra flash-attn
