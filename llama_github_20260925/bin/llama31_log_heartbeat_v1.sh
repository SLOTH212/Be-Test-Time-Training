#!/usr/bin/env bash
set -uo pipefail
R=/path/to/ttt
exec >> "$R/logs/llama31_derived_data_v1/master.log" 2>&1
for _ in $(seq 1 30); do
  pid="$(cat "$R/work/llama31_derived_data_v1/controller.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 1
done
if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then exit 1; fi
while kill -0 "$pid" 2>/dev/null; do
  phase="$($R/envs/ttt_runtime_v1/bin/python -c 'import json; print(json.load(open("/path/to/ttt/work/llama31_derived_data_v1/controller_state.json")).get("phase","unknown"))' 2>/dev/null || echo unknown)"
  echo "HEARTBEAT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$pid phase=$phase"
  sleep 30
done
