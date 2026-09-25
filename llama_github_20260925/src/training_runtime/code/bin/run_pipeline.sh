#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=${TTT_TRAINING_ROOT}
exec 9>"$PIPELINE_ROOT/locks/pipeline.lock"
flock -n 9 || { echo BLOCKED_PIPELINE_LOCKED >&2; exit 75; }
rm -f "$PIPELINE_ROOT/locks/CANCEL_AUTO_SHUTDOWN"
AUTO_SHUTDOWN_ENABLED=$(${TTT_PYTHON} -c 'import yaml; print(str(yaml.safe_load(open(__import__("os").environ["TTT_TRAINING_ROOT"]+"/configs/pipeline_runtime.yaml"))["auto_shutdown"]["enabled"]).lower())')
AUTO_SHUTDOWN_ON_SUCCESS=$(${TTT_PYTHON} -c 'import yaml; print(str(yaml.safe_load(open(__import__("os").environ["TTT_TRAINING_ROOT"]+"/configs/pipeline_runtime.yaml"))["auto_shutdown"]["on_success"]).lower())')
AUTO_SHUTDOWN_ON_FAILURE=$(${TTT_PYTHON} -c 'import yaml; print(str(yaml.safe_load(open(__import__("os").environ["TTT_TRAINING_ROOT"]+"/configs/pipeline_runtime.yaml"))["auto_shutdown"]["on_failure"]).lower())')
AUTO_SHUTDOWN_DELAY_SECONDS=$(${TTT_PYTHON} -c 'import yaml; print(yaml.safe_load(open(__import__("os").environ["TTT_TRAINING_ROOT"]+"/configs/pipeline_runtime.yaml"))["auto_shutdown"]["delay_seconds"])')
AUTO_SHUTDOWN_DRY_RUN=$(${TTT_PYTHON} -c 'import yaml; print(str(yaml.safe_load(open(__import__("os").environ["TTT_TRAINING_ROOT"]+"/configs/pipeline_runtime.yaml"))["auto_shutdown"]["dry_run"]).lower())')
export AUTO_SHUTDOWN_ENABLED AUTO_SHUTDOWN_ON_SUCCESS AUTO_SHUTDOWN_ON_FAILURE AUTO_SHUTDOWN_DELAY_SECONDS AUTO_SHUTDOWN_DRY_RUN
printf 'AUTO_SHUTDOWN_ENABLED=%s\nAUTO_SHUTDOWN_ON_SUCCESS=%s\nAUTO_SHUTDOWN_ON_FAILURE=%s\nAUTO_SHUTDOWN_DELAY_SECONDS=%s\nAUTO_SHUTDOWN_DRY_RUN=%s\n' "$AUTO_SHUTDOWN_ENABLED" "$AUTO_SHUTDOWN_ON_SUCCESS" "$AUTO_SHUTDOWN_ON_FAILURE" "$AUTO_SHUTDOWN_DELAY_SECONDS" "$AUTO_SHUTDOWN_DRY_RUN"
bash "$PIPELINE_ROOT/bin/preflight.sh" --formal
${TTT_PYTHON} "$PIPELINE_ROOT/bin/pipelinectl.py" run "${1:-}"
