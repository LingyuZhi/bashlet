#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
experiment_config="${1:-configs/opsd.yaml}"
if (( $# )); then shift; fi
exec uv run --locked bashlet train-opsd "$experiment_config" "$@"
