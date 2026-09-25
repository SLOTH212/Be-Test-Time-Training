#!/usr/bin/env bash
set -Eeuo pipefail
PIPELINE_ROOT=/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2
exec 9>"$PIPELINE_ROOT/locks/pipeline.lock"
flock -n 9 || { echo BLOCKED_PIPELINE_LOCKED >&2; exit 75; }
rm -f "$PIPELINE_ROOT/locks/CANCEL_AUTO_SHUTDOWN"
AUTO_SHUTDOWN_ENABLED=$(/root/miniconda3/envs/inplace_ttt_autodl/bin/python -c 'import yaml; print(str(yaml.safe_load(open("/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2/configs/pipeline_runtime.yaml"))["auto_shutdown"]["enabled"]).lower())')
AUTO_SHUTDOWN_ON_SUCCESS=$(/root/miniconda3/envs/inplace_ttt_autodl/bin/python -c 'import yaml; print(str(yaml.safe_load(open("/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2/configs/pipeline_runtime.yaml"))["auto_shutdown"]["on_success"]).lower())')
AUTO_SHUTDOWN_ON_FAILURE=$(/root/miniconda3/envs/inplace_ttt_autodl/bin/python -c 'import yaml; print(str(yaml.safe_load(open("/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2/configs/pipeline_runtime.yaml"))["auto_shutdown"]["on_failure"]).lower())')
AUTO_SHUTDOWN_DELAY_SECONDS=$(/root/miniconda3/envs/inplace_ttt_autodl/bin/python -c 'import yaml; print(yaml.safe_load(open("/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2/configs/pipeline_runtime.yaml"))["auto_shutdown"]["delay_seconds"])')
AUTO_SHUTDOWN_DRY_RUN=$(/root/miniconda3/envs/inplace_ttt_autodl/bin/python -c 'import yaml; print(str(yaml.safe_load(open("/root/autodl-tmp/ttt/pipelines/gated_ntp_300m_qa10m_eval13k_v2/configs/pipeline_runtime.yaml"))["auto_shutdown"]["dry_run"]).lower())')
export AUTO_SHUTDOWN_ENABLED AUTO_SHUTDOWN_ON_SUCCESS AUTO_SHUTDOWN_ON_FAILURE AUTO_SHUTDOWN_DELAY_SECONDS AUTO_SHUTDOWN_DRY_RUN
printf 'AUTO_SHUTDOWN_ENABLED=%s\nAUTO_SHUTDOWN_ON_SUCCESS=%s\nAUTO_SHUTDOWN_ON_FAILURE=%s\nAUTO_SHUTDOWN_DELAY_SECONDS=%s\nAUTO_SHUTDOWN_DRY_RUN=%s\n' "$AUTO_SHUTDOWN_ENABLED" "$AUTO_SHUTDOWN_ON_SUCCESS" "$AUTO_SHUTDOWN_ON_FAILURE" "$AUTO_SHUTDOWN_DELAY_SECONDS" "$AUTO_SHUTDOWN_DRY_RUN"
bash "$PIPELINE_ROOT/bin/preflight.sh" --formal
/root/miniconda3/envs/inplace_ttt_autodl/bin/python "$PIPELINE_ROOT/bin/pipelinectl.py" run "${1:-}"
