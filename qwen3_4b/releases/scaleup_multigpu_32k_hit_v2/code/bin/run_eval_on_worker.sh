#!/usr/bin/env bash
set -Eeuo pipefail
R=/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2
P=/root/autodl-tmp/ttt/inplace_ttt_gated_ntp_perf_parity_v1
cd "$P"
exec /root/miniconda3/envs/inplace_ttt_autodl/bin/python "$R/workers/eval_worker.py" --mode on "$@"
