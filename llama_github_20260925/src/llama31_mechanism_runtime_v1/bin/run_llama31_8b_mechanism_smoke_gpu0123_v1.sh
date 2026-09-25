#!/usr/bin/env bash
set -euo pipefail
source /path/to/ttt/bin/activate_ttt.sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../cli.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  ROOT="${LLAMA_MECHANISM_ROOT:-/path/to/ttt/src/llama31_mechanism_runtime_v1}"
fi
exec python -B "$ROOT/cli.py" smoke "$@"
