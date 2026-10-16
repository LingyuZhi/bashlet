#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run --locked bashlet train-opsd \
  configs/opsd-expanded-v1.yaml \
  --preflight-config configs/opsd-expanded-preflight-v1.yaml \
  --execution-limit configs/opsd-expanded-execution-536-v1.yaml \
  --output-dir runs/issue32-four-epoch-20261009 "$@"
