#!/usr/bin/env bash
set -Eeuo pipefail
R=${TTT_TRAINING_ROOT}
P=${TTT_PROJECT_ROOT}
cd "$P"
exec ${TTT_PYTHON} "$R/workers/eval_worker.py" --mode on "$@"
