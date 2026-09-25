#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/USER/ttt/runs/formal/exact_reverse_1p7b_v1/run_20260905T144633+0800
PYTHON_BIN=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
WORKER_SCRIPT="$RUN_ROOT/scripts/exact_reverse_worker.py"
AGGREGATE_SCRIPT="$RUN_ROOT/scripts/aggregate_exact_reverse.py"
CONTROLLER_LOG="$RUN_ROOT/logs/controller.log"
LOCK_FILE="$RUN_ROOT/status/controller.lock"

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "CONTROLLER_ALREADY_RUNNING" >&2; exit 75; }

printf '%s\n' "$$" > "$RUN_ROOT/status/controller.pid.tmp"
mv "$RUN_ROOT/status/controller.pid.tmp" "$RUN_ROOT/status/controller.pid"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export EXACT_REVERSE_CONTROLLER_PID="$$"

{
  echo "PHASE_0=AUTHORITY_VERIFICATION"
  echo "PHASE_1=PROTOCOL_AMENDMENT_VERIFICATION"
  echo "PHASE_2=COHORT_MANIFEST_VERIFICATION"
  echo "PHASE_3=SMOKE_PARITY_GATE_VERIFICATION"
  echo "PHASE_4=FORMAL_173_EXACT_REVERSE_EXECUTION"
} >> "$CONTROLLER_LOG"

taskset -c 8-23 "$PYTHON_BIN" "$WORKER_SCRIPT" "$RUN_ROOT" --mode formal >> "$CONTROLLER_LOG" 2>&1

{
  echo "PHASE_5=COMPLETENESS_INTEGRITY_AUDIT"
  echo "PHASE_6=AGGREGATION"
  echo "PHASE_7=PAIRED_BOOTSTRAP"
  echo "PHASE_8=TASK_STRATIFIED_PAIRED_BOOTSTRAP"
  echo "PHASE_9=PER_TASK_REPORT"
  echo "PHASE_10=FINAL_COMPLETION_RECEIPT"
} >> "$CONTROLLER_LOG"

taskset -c 8-23 "$PYTHON_BIN" "$AGGREGATE_SCRIPT" "$RUN_ROOT" >> "$CONTROLLER_LOG" 2>&1
echo "CONTROLLER_COMPLETE=true" >> "$CONTROLLER_LOG"

