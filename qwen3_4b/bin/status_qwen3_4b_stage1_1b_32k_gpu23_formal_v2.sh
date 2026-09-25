#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/path/to/ttt
source "$ROOT/bin/activate_ttt.sh"
export CUDA_VISIBLE_DEVICES=2,3 WORLD_SIZE=2 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/tmp" OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$TTT_PYTHON" -B "$ROOT/bin/qwen3_4b_stage1_gpu23_formal_control_v2.py" status
