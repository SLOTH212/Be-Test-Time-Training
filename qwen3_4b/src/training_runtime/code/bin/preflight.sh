#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
exec ${TTT_PYTHON} "$PIPELINE_ROOT/bin/preflight.py" "$@"

