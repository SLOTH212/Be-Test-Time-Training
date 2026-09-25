#!/usr/bin/env bash
set -Eeuo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
exec python -B /path/to/ttt/bin/qwen3_4b_stage2_formal_control_v1.py status
