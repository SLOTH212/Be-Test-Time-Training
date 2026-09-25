#!/usr/bin/env bash
set -euo pipefail
R=/path/to/ttt
status="$($R/envs/ttt_runtime_v1/bin/python -c 'import json; print(json.load(open("/path/to/ttt/work/llama31_derived_data_v1/PRELAUNCH.json"))["status"])')"
test "$status" = PASS
if tmux has-session -t llama31_data_v1 2>/dev/null; then
  echo SESSION_ALREADY_EXISTS
  exit 2
fi
tmux new-session -d -s llama31_data_v1 "$R/bin/build_llama31_derived_data_v1.sh >> $R/logs/llama31_derived_data_v1/master.log 2>&1"
tmux new-window -d -t llama31_data_v1 -n progress /bin/bash "$R/bin/llama31_log_heartbeat_v1.sh"
echo LAUNCHED
