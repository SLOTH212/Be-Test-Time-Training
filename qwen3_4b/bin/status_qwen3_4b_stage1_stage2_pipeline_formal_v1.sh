#!/usr/bin/env bash
set -Eeuo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
exec "/path/to/ttt/envs/ttt_runtime_v1/bin/python" -B /path/to/ttt/bin/qwen3_4b_stage1_stage2_pipeline_formal_v1.py --status "$@"
