#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 SOURCE FINAL" >&2; exit 64; }
src=$1; dst=$2
[[ -d "$src" ]] || exit 65
[[ ! -e "$dst" ]] || { echo "refuse to overwrite final artifact" >&2; exit 66; }
partial="${dst}.partial"
[[ ! -e "$partial" ]] || { echo "partial promotion exists" >&2; exit 67; }
cp -al "$src" "$partial"
find "$partial" -type f -print0 | sort -z | xargs -0 sha256sum > "$partial/SHA256SUMS.txt"
sync
mv "$partial" "$dst"

