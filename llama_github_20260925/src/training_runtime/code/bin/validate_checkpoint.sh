#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 CHECKPOINT_DIR" >&2; exit 64; }
dir=$1
[[ -f "$dir/checkpoint_manifest.json" ]] || { echo missing_manifest >&2; exit 65; }
(cd "$dir" && sha256sum -c SHA256SUMS.txt)

