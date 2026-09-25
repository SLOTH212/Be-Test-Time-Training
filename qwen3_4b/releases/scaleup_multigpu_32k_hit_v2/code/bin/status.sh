#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2
for f in "$PIPELINE_ROOT"/state/*.json; do
  [[ -e "$f" ]] || continue
  echo "== $f"
  /root/miniconda3/envs/inplace_ttt_autodl/bin/python -m json.tool "$f"
done

