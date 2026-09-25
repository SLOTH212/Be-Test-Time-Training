#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2
exec /root/miniconda3/envs/inplace_ttt_autodl/bin/python "$PIPELINE_ROOT/bin/preflight.py" "$@"

