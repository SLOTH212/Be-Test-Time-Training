#!/usr/bin/env bash
set -Eeuo pipefail
R=${TTT_TRAINING_ROOT}
mode=${1:?mode required}
case "$mode" in
  off) exec bash "$R/bin/run_eval_off_worker.sh" --config "$R/configs/eval_off_13k.yaml" --common-config "$R/configs/eval_common_13k.yaml" --output-dir "$R/inference/off" --state-file "$R/state/eval_off_state.json" --checkpoint "$R/artifacts/stage2_final_eval" --formal-run ;;
  on) exec bash "$R/bin/run_eval_on_worker.sh" --config "$R/configs/eval_on_13k.yaml" --common-config "$R/configs/eval_common_13k.yaml" --output-dir "$R/inference/on" --state-file "$R/state/eval_on_state.json" --checkpoint "$R/artifacts/stage2_final_eval" --formal-run ;;
  *) echo "invalid mode: $mode" >&2; exit 64 ;;
esac
