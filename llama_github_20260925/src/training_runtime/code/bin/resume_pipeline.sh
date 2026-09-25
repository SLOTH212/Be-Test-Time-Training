#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
exec bash "$PIPELINE_ROOT/bin/run_pipeline.sh" --resume

