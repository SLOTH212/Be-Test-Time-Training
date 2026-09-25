#!/usr/bin/env bash
set -uo pipefail
R=/path/to/ttt
W="$R/work/llama31_derived_data_v1"
STATE="$W/controller_state.json"
PIDFILE="$W/controller.pid"
pid="$(cat "$PIDFILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then alive=alive; else alive=dead; fi
echo "controller=$alive pid=${pid:-unknown}"
if [[ -f "$STATE" ]]; then "$R/envs/ttt_runtime_v1/bin/python" -c 'import json,sys; x=json.load(open(sys.argv[1])); print("phase="+str(x.get("phase","unknown"))); print("status="+str(x.get("status","unknown"))); print("elapsed_seconds="+str(round(__import__("time").time()-x.get("started",__import__("time").time()),1))); print("latest_error="+str(x.get("latest_error","none")))' "$STATE"; fi
for split in train validation; do
  s="$W/stage1_${split}_state.json"
  if [[ -f "$s" ]]; then "$R/envs/ttt_runtime_v1/bin/python" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(sys.argv[2]+"_selected_units="+str(x.get("next",0))); print(sys.argv[2]+"_records="+str(x.get("records",0))); print(sys.argv[2]+"_input_tokens="+str(x.get("tokens",0)))' "$s" "stage1_$split"; fi
done
if [[ -f "$R/datasets/llama31_stage2_15m_32k_v1/audits/LLAMA_STAGE2_MATERIALIZATION.json" ]]; then "$R/envs/ttt_runtime_v1/bin/python" -c 'import json,sys; x=json.load(open(sys.argv[1])); print("stage2_records="+str(x.get("records",0))); print("stage2_input_tokens="+str(x.get("input_tokens",0))); print("stage2_qa_targets="+str(x.get("qa_targets",0)))' "$R/datasets/llama31_stage2_15m_32k_v1/audits/LLAMA_STAGE2_MATERIALIZATION.json"; fi
for a in "$R/datasets/llama31_stage1_500m_32k_v1/audits/LLAMA_STAGE1_INDEPENDENT_AUDIT.json" "$R/datasets/llama31_stage2_15m_32k_v1/audits/LLAMA_STAGE2_INDEPENDENT_AUDIT.json"; do [[ -f "$a" ]] && "$R/envs/ttt_runtime_v1/bin/python" -c 'import json,sys; print("audit="+sys.argv[1]+" status="+str(json.load(open(sys.argv[1])).get("status")))' "$a"; done
du -sh "$W" "$R/datasets/llama31_stage1_500m_32k_v1" "$R/datasets/llama31_stage2_15m_32k_v1" 2>/dev/null || true
