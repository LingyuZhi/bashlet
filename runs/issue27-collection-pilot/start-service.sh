#!/usr/bin/env bash
set -euo pipefail
cd /data/home/zdhs0148/opd/bashlet
module avail
module load cuda/12.8
date -u +%FT%TZ > runs/issue27-collection-pilot/service-started-at.txt
nvidia-smi > runs/issue27-collection-pilot/gpu-before.txt
exec timeout --signal=TERM --kill-after=45s 6900s \
  bash scripts/serve-local-vllm.sh configs/local-vllm/serve-collection-pilot-v1.yaml
