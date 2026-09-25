#!/usr/bin/env bash
set -Eeuo pipefail
R=${TTT_TRAINING_ROOT}
exec bash "$R/bin/run_stage1_worker.sh" \
  --config "$R/configs/stage1_300m.yaml" --output-dir "$R" \
  --state-file "$R/state/stage1_state.json" --resume-from "$R/checkpoints/stage1" --formal-run
