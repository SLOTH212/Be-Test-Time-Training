#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800
PYTHON_BIN=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
WORKER="$RUN_ROOT/scripts/exact_r2_worker.py"
AGGREGATOR="$RUN_ROOT/scripts/aggregate_exact_r2.py"
LOG="$RUN_ROOT/logs/controller.log"
LOCK="$RUN_ROOT/status/controller.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "CONTROLLER_ALREADY_RUNNING" >&2; exit 75; }
printf '%s\n' "$$" > "$RUN_ROOT/status/controller.pid.tmp"
mv "$RUN_ROOT/status/controller.pid.tmp" "$RUN_ROOT/status/controller.pid"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
export EXACT_R2_CONTROLLER_PID="$$"
{
  echo "PHASE_0=AUTHORITY_HASH_VERIFICATION"
  echo "PHASE_1=PREREGISTRATION_VERIFICATION"
  echo "PHASE_2=CANDIDATE_MANIFEST_AUDIT"
  echo "PHASE_3=ONE_SWITCH_MAPPING_AUDIT"
  echo "PHASE_4_5=SMOKE_GATE_VERIFICATION"
  echo "PHASE_6=FORMAL_99120_CANDIDATE_EXECUTION"
} >> "$LOG"
taskset -c 8-23 "$PYTHON_BIN" "$WORKER" "$RUN_ROOT" --mode formal >> "$LOG" 2>&1
{
  echo "PHASE_7=CANDIDATE_INTEGRITY_AUDIT"
  echo "PHASE_8=WINNER_SELECTION"
  echo "PHASE_9=INDEPENDENT_WINNER_AUDIT"
  echo "PHASE_10=ONE_SWITCH_INVARIANT"
  echo "PHASE_11_15=AGGREGATION_BOOTSTRAP_SECONDARY_DESCRIPTIVE"
  echo "PHASE_16=FINAL_COMPLETION_RECEIPT"
} >> "$LOG"
taskset -c 8-23 "$PYTHON_BIN" "$AGGREGATOR" "$RUN_ROOT" >> "$LOG" 2>&1
echo "CONTROLLER_COMPLETE=true" >> "$LOG"
