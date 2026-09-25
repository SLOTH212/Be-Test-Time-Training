#!/usr/bin/env bash
set -Eeuo pipefail
R=/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2
exec bash "$R/bin/run_stage2_worker.sh" \
  --config "$R/configs/stage2_qa10m.yaml" --output-dir "$R" \
  --state-file "$R/state/stage2_state.json" --resume-from "$R/checkpoints/stage2" \
  --parent-artifact "$R/artifacts/stage1_final_resume" --formal-run
