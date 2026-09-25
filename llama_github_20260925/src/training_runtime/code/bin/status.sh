#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
for f in "$PIPELINE_ROOT"/state/*.json; do
  [[ -e "$f" ]] || continue
  echo "== $f"
  ${TTT_PYTHON} -m json.tool "$f"
done

