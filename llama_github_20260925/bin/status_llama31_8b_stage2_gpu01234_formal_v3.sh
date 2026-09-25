#!/usr/bin/env bash
set -euo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 WORLD_SIZE=5
exec python -B /path/to/ttt/work/llama31_8b_stage2_gpu01234_formal_v3/pipeline.py status "$@"
