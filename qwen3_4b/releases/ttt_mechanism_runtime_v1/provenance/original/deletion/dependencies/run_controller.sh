#!/usr/bin/env bash
set -euo pipefail
RUN=/home/USER/ttt/runs/formal/selective_deletion_1p7b_v1/run_20260905T201047+0800
PY=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
exec 9>"$RUN/status/controller.lock"; flock -n 9 || { echo CONTROLLER_ALREADY_RUNNING >&2; exit 75; }
printf '%s\n' "$$" > "$RUN/status/controller.pid.tmp"; mv "$RUN/status/controller.pid.tmp" "$RUN/status/controller.pid"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 SELECTIVE_CONTROLLER_PID="$$"
for p in 0 1 2 3 4 5; do echo "PHASE_$p=PRELAUNCH_GATE" >> "$RUN/logs/controller.log"; done
echo PHASE_6=FORMAL_458_BRANCH_EXECUTION >> "$RUN/logs/controller.log"
taskset -c 8-23 "$PY" "$RUN/scripts/selective_worker.py" "$RUN" --mode formal >> "$RUN/logs/controller.log" 2>&1
for p in 7 8 9 10 11 12 13 14; do echo "PHASE_$p=POSTEXECUTION" >> "$RUN/logs/controller.log"; done
taskset -c 8-23 "$PY" "$RUN/scripts/aggregate_selective.py" "$RUN" >> "$RUN/logs/controller.log" 2>&1
echo CONTROLLER_COMPLETE=true >> "$RUN/logs/controller.log"
