#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
export AUTO_SHUTDOWN_DRY_RUN=true
exec ${TTT_PYTHON} "$PIPELINE_ROOT/tests/run_tests.py"

