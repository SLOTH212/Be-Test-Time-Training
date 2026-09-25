#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" "$ROOT/workers/distributed_train_worker.py" \
  --stage 2 --config "$ROOT/configs/stage2_qa_multigpu_v1.yaml" --output-dir "$ROOT/logs/stage2_multigpu" \
  --parent-artifact "$ROOT/artifacts/stage1_model_only_dcp" "$@"
