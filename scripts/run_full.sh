#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync python -m grkv.run --config configs/mistral.yaml --stage full --fresh "$@"
uv run --no-sync python -m grkv.run --config configs/llama.yaml --stage full --fresh "$@"
