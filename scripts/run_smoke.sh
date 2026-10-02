#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync python -m grkv.run --config configs/mistral_k1_g000.yaml --stage smoke "$@"
uv run --no-sync python -m grkv.run --config configs/llama_k1_g010.yaml --stage smoke "$@"
