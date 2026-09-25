#!/usr/bin/env bash
set -u -o pipefail
RUN=/home/USER/ttt/runs/formal/kp_factorial_1p7b_v1/run_20260905T235503+0800
PY=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
LOG="$RUN/logs/controller.log"
exec > >(tee -a "$LOG") 2>&1
echo "CONTROLLER_START=$(date --iso-8601=seconds) PID=$$ RUN_ROOT=$RUN"
attempt=1
while [ "$attempt" -le 3 ]; do
  echo "FORMAL_ATTEMPT=$attempt"
  if taskset -c 8-23 "$PY" "$RUN/scripts/kp_worker.py" "$RUN" --mode formal; then
    taskset -c 8-23 "$PY" "$RUN/scripts/aggregate_kp.py" "$RUN"
    echo "CONTROLLER_COMPLETE=$(date --iso-8601=seconds)"
    exit 0
  fi
  if grep -Eq 'AUTHORITY_HASH_MISMATCH|NATIVE_REFERENCE_REPLAY_PARITY|FACTOR_ISOLATION_FAILURE|F11_NONCLOSURE|BASE_PARAMETER_MUTATION|FOUR_CELL_COMPLETENESS' "$LOG"; then
    echo "SCIENTIFIC_FAIL_FAST=true"
    exit 1
  fi
  attempt=$((attempt+1))
done
echo "FINITE_RETRY_EXHAUSTED=true"
exit 1
