#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/USER/ttt/runs/formal/state_reset_1p7b_v1/run_20260905T180918+0800
PY=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
LOG="$RUN_ROOT/logs/controller.log"
exec 9>"$RUN_ROOT/status/controller.lock"
flock -n 9 || { echo CONTROLLER_ALREADY_RUNNING >&2; exit 75; }
printf '%s\n' "$$" > "$RUN_ROOT/status/controller.pid.tmp"; mv "$RUN_ROOT/status/controller.pid.tmp" "$RUN_ROOT/status/controller.pid"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 STATE_RESET_CONTROLLER_PID="$$"
for p in 0 1 2 3 4 5; do echo "PHASE_$p=PRELAUNCH_GATE" >> "$LOG"; done
echo PHASE_6=FORMAL_167_STATE_RESET_EXECUTION >> "$LOG"
taskset -c 8-23 "$PY" "$RUN_ROOT/scripts/state_reset_worker.py" "$RUN_ROOT" --mode formal >> "$LOG" 2>&1
for p in 7 8 9 10 11 12; do echo "PHASE_$p=POST_EXECUTION_AGGREGATION" >> "$LOG"; done
taskset -c 8-23 "$PY" "$RUN_ROOT/scripts/aggregate_state_reset.py" "$RUN_ROOT" >> "$LOG" 2>&1
echo CONTROLLER_COMPLETE=true >> "$LOG"
