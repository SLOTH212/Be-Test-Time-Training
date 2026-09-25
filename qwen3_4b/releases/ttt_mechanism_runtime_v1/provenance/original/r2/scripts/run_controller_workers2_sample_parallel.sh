#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800
PYTHON_BIN=/home/USER/conda_envs/ttt_phase_c_v1/bin/python
exec "$PYTHON_BIN" "$RUN_ROOT/scripts/controller_workers2_sample_parallel.py" >> "$RUN_ROOT/logs/controller_workers2_sample_parallel.log" 2>&1
