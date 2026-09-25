#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
CANCEL_FILE="$PIPELINE_ROOT/locks/CANCEL_AUTO_SHUTDOWN"
[[ $(id -u) -eq 0 ]] || exit 77
[[ "${PIPELINE_MASTER_PROCESS:-}" == "1" ]] || exit 78
[[ "${PIPELINE_EXECUTION_MODE:-}" == "formal" ]] || exit 79
[[ "${AUTO_SHUTDOWN_ENABLED:-false}" == "true" ]] || exit 80
[[ "${AUTO_SHUTDOWN_ON_SUCCESS:-false}" == "true" ]] || exit 81
[[ "${AUTO_SHUTDOWN_ON_FAILURE:-true}" == "false" ]] || exit 82
[[ "${AUTO_SHUTDOWN_DRY_RUN:-true}" == "false" ]] || { echo ACTUAL_SHUTDOWN_TRIGGERED=false; exit 0; }
[[ "${AUTO_SHUTDOWN_DELAY_SECONDS:-0}" == "300" ]] || exit 83
${TTT_PYTHON} "$PIPELINE_ROOT/bin/pipelinectl.py" shutdown-gate
echo AUTO_SHUTDOWN_SCHEDULED
echo AUTO_SHUTDOWN_DELAY_SECONDS=300
echo "CANCEL_COMMAND=touch $CANCEL_FILE"
sync
sleep 5
sync
for ((elapsed=0; elapsed<AUTO_SHUTDOWN_DELAY_SECONDS; elapsed+=10)); do
  if [[ -e "$CANCEL_FILE" ]]; then
    echo AUTO_SHUTDOWN_CANCELLED=true
    echo AUTO_SHUTDOWN_EXECUTED=false
    echo INSTANCE_LEFT_RUNNING=true
    exit 0
  fi
  sleep 10
done
if command -v systemctl >/dev/null 2>&1; then
  systemctl poweroff
elif command -v shutdown >/dev/null 2>&1; then
  shutdown -h now
elif command -v poweroff >/dev/null 2>&1; then
  poweroff
else
  exit 127
fi
