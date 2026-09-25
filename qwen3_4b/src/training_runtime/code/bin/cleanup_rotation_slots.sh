#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
[[ "${1:-}" == "--execute" ]] || { echo "dry-run: slot_A/slot_B would be removed only after FINAL_CLEANUP validation"; exit 0; }
exec ${TTT_PYTHON} "$PIPELINE_ROOT/bin/pipelinectl.py" cleanup

