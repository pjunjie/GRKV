#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage smoke "$@"
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage smoke "$@"
