#!/usr/bin/env bash
set -euo pipefail
# Invoke from the repository root, after loading the recorded CUDA module.
cd "$(dirname "$0")/.."
service_config="${1:-runs/local-vllm/serve.yaml}"
if (( $# == 0 )); then
  uv run --locked bashlet write-vllm-config configs/opsd.yaml --output "$service_config"
fi
exec uv run --env-file environments/local-vllm/server.env --project environments/local-vllm --locked \
  python -m vllm.entrypoints.cli.main serve --config "$service_config"
