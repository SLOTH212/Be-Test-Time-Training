#!/usr/bin/env bash
set -euo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 WORLD_SIZE=4
exec python -B /path/to/ttt/work/llama31_8b_stage1_stage2_pipeline_formal_v1/pipeline.py finalize --stage 2 "$@"
