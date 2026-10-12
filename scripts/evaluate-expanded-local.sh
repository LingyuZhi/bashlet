#!/usr/bin/env bash
set -euo pipefail
exec uv run --locked bashlet evaluate-checkpoint configs/opsd-expanded-local-eval-v1.yaml "$@"
