#!/usr/bin/env bash
set -euo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python /path/to/ttt/src/qwen3_4b_downstream_pipeline_v1/formal_control.py status --phase mechanism "$@"
