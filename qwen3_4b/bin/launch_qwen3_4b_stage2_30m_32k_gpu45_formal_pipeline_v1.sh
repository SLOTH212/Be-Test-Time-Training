#!/usr/bin/env bash
set -Eeuo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export CUDA_VISIBLE_DEVICES=4,5 WORLD_SIZE=2 PYTHONDONTWRITEBYTECODE=1
unset CUBLAS_WORKSPACE_CONFIG PYTHONHASHSEED
exec python -B /path/to/ttt/bin/qwen3_4b_stage2_gpu45_formal_control_pipeline_v1.py launch "$@"
