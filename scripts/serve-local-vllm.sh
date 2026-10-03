#!/usr/bin/env bash
set -euo pipefail
# Invoke from the repository root, after loading the recorded CUDA module.
exec uv run --env-file configs/local-vllm/server.env --project configs/local-vllm --locked vllm serve \
  --config "${1:-configs/local-vllm/serve.yaml}"
