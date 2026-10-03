#!/usr/bin/env bash
set -euo pipefail
cd /data/home/zdhs0148/opd/bashlet
module avail
module load cuda/12.8
uv sync --project configs/local-vllm --locked
setsid nohup bash scripts/serve-local-vllm.sh "${1:-configs/local-vllm/serve.yaml}" \
  > "runs/issue21-local-inference/server-${2:-base}.log" 2>&1 < /dev/null &
echo "$!" > runs/issue21-local-inference/service.pid
