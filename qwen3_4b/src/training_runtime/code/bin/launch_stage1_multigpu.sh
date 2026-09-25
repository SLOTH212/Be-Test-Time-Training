#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" "$ROOT/workers/distributed_train_worker.py" \
  --stage 1 --config "$ROOT/configs/stage1_1b_32k_multigpu_v1.yaml" --output-dir "$ROOT/logs/stage1_multigpu" "$@"
