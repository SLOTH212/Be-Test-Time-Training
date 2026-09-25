#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/path/to/ttt/cache/huggingface
export XDG_CACHE_HOME=/path/to/ttt/cache/xdg
mkdir -p /path/to/ttt/work/llama31_derived_data_v1 /path/to/ttt/logs/llama31_derived_data_v1
echo "$$" > /path/to/ttt/work/llama31_derived_data_v1/controller.pid.tmp
mv /path/to/ttt/work/llama31_derived_data_v1/controller.pid.tmp /path/to/ttt/work/llama31_derived_data_v1/controller.pid
exec /path/to/ttt/envs/ttt_runtime_v1/bin/python /path/to/ttt/src/llama31_data_derivation_v1/llama31_derive.py controller
