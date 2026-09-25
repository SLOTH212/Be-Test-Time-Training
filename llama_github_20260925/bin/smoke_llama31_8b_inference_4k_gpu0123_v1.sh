#!/usr/bin/env bash
set -euo pipefail
source /path/to/ttt/bin/activate_ttt.sh
/path/to/ttt/bin/check_shared_assets.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID WORLD_SIZE=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -B /path/to/ttt/work/llama31_8b_inference_smoke_v1/inference_control.py "$@"
