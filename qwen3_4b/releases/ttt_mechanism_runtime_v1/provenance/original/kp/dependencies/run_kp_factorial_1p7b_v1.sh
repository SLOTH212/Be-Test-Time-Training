#!/usr/bin/env bash
set -euo pipefail
RUN=/home/USER/ttt/runs/formal/kp_factorial_1p7b_v1/run_20260905T235503+0800
TMUX=/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/tools/tmux
SESSION=kp_factorial_1p7b_v1
if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_SESSION_ALREADY_EXISTS=$SESSION"
  exit 0
fi
"$TMUX" new-session -d -s "$SESSION" "bash '$RUN/scripts/run_kp_controller.sh'"
echo "FORMAL_LAUNCHED=true"
echo "RUN_ROOT=$RUN"
echo "TMUX_SESSION=$SESSION"
