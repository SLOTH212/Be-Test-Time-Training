import pathlib,json,hashlib,yaml,time
R=pathlib.Path("/path/to/ttt");A=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_determinism_v4"
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest_path=A/"PRELAUNCH_AUDIT_INPUT_MANIFEST.json";manifest=load(manifest_path)
review_path=A/"FINAL_INDEPENDENT_PRELAUNCH_AUDIT.json";review=load(review_path)
assert manifest["status"]=="READY_FOR_INDEPENDENT_REVIEW"
assert review["status"]=="PASS" and review["reviewed_input_manifest_sha256"]==sha(manifest_path)
assert review["auditor_agent"]=="independent_stop_audit" and not review.get("blocking_findings")
files=manifest["file_sha256"]
for p,h in files.items():assert sha(p)==h,("FINAL_FILE_IDENTITY_CHANGED",p)
for p in [manifest_path,review_path]:files[str(p)]=sha(p)
cfgpath=R/"configs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1.yaml";cfg=yaml.safe_load(cfgpath.read_text())
assert all(cfg[k] for k in ["capacity_parameters_frozen","formal_gpu_policy_frozen","formal_batch_semantics_frozen"])
assert cfg["world_size"]==2 and cfg["physical_gpu_allowlist"]=="2,3" and cfg["micro_batch_size"]==cfg["gradient_accumulation"]==1 and cfg["global_batch_size"]==2
assert not pathlib.Path(cfg["output_root"]).exists() and not pathlib.Path(cfg["stage1_model_only_artifact"]).exists()
g=load(A/"REAL_RUNTIME_GATES.json");assert g["status"]=="PASS"
t=load(A/"STEADY_STATE_THROUGHPUT.json");assert t["status"]=="PASS" and t["warmup_optimizer_steps"]>=5 and t["measured_optimizer_steps"]>=20
for p in ["REAL_DCP_AND_RESUME_RNG.json","REAL_RESUME_NUMERIC_PARITY.json","REAL_FAILURE_RECOVERY_NUMERIC_PARITY.json"]:assert load(A/p)["status"]=="PASS"
estimate=t["wall_time_estimate"];policy=load(R/"config/CHECKPOINT_RETENTION_POLICY.json")
authority={
"schema":"QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V1",
"QWEN3_4B_STAGE1_PRELAUNCH_STATUS":"PASS_WITH_WARNING",
"SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23":"YES",
"HOST":"amax","USER":"USER",
"TRAINING_RUNTIME_AUTHORITY":"PASS","TRAINING_RUNTIME_SHA256":"79d89c95908ea787c1358a651c9003ca14eced1957853154b43041db845c162b",
"ADAPTED_RUNTIME_SOURCE_MANIFEST_SHA256":sha(cfg["runtime_source_manifest"]),
"SHARED_ASSET_GUARD":"PASS",
"MODEL_PATH":cfg["model_base"],"MODEL_AUTHORITY_STATUS":"PASS","QWEN3_4B_MODEL_AUTHORITY":"PASS","MODEL_PARAMETER_COUNT":4061881856,"MODEL_NUM_HIDDEN_LAYERS":36,
"STAGE1_DATASET_ROOT":str(R/"datasets"),"STAGE1_DATASET_RUNTIME_AUTHORITY":"PASS","STAGE1_PACKAGE_SHA256":"11ddc46d95381c6bdaed7f3a62a7f2f2711e95db1223f7f50ec85e249832217d",
"TRAIN_LOGICAL_TOKENS":990000000,"VAL_LOGICAL_TOKENS":10000000,"TOTAL_LOGICAL_TOKENS":1000000000,"TRAIN_ACTUAL_INPUT_TOKENS":989996971,"VAL_ACTUAL_INPUT_TOKENS":9999975,
"CONTEXT_LENGTH":32768,"TTT_CHUNK_SIZE":4096,"TTT_LAYERS":[0,6,12,18,24,30],
"PHYSICAL_GPU_ALLOWLIST":[2,3],"FORMAL_GPU_SET":[2,3],"FORMAL_WORLD_SIZE":2,"CUDA_VISIBLE_DEVICE_COUNT":2,"GPU_MAPPING_STATUS":"PASS","UNAUTHORIZED_GPU_USED":False,
"GPU2_STATUS":"AVAILABLE_WITH_USER_AUTHORIZED_EXISTING_PROCESSES","GPU3_STATUS":"AVAILABLE_WITH_USER_AUTHORIZED_EXISTING_PROCESSES",
"REAL_STAGE1_DATALOADER_PREFLIGHT":"PASS","HISTORICAL_BATCH_SEMANTICS_STATUS":"PASS","historical_batch":load(R/"audits/BATCH_SEMANTICS_AUDIT.json"),
"FORMAL_MICRO_BATCH_PER_GPU":1,"FORMAL_GRAD_ACCUM":1,"FORMAL_GLOBAL_BATCH":2,"FORMAL_LOGICAL_TOKENS_PER_UPDATE":cfg["logical_tokens_per_update"],"FORMAL_INPUT_TOKENS_PER_UPDATE":cfg["input_tokens_per_update"],
"FORMAL_BATCH_SEMANTICS_FROZEN":True,"FORMAL_GPU_POLICY_FROZEN":True,
**{k:v for k,v in g.items() if k.startswith(("REAL_DATA_","FORMAL_TTT_","FORMAL_CHUNK4K_","STAGE1_VALIDATION_"))},
"STEADY_STATE_STEP_TIME_MEAN":t["step_seconds"]["mean"],"STEADY_STATE_STEP_TIME_P50":t["step_seconds"]["median"],"STEADY_STATE_STEP_TIME_P95":t["step_seconds"]["p95"],
"STEADY_STATE_INPUT_TOKENS_PER_SEC":t["input_tokens_per_second"],"STEADY_STATE_LOGICAL_TOKENS_PER_SEC":"No exact source-logical-budget per-record mapping; use measured actual input token rate with explicit counter semantics.",
"DATALOADER_WAIT_FRACTION":t["dataloader_wait_fraction"],
"ESTIMATED_1B_STAGE1_WALL_TIME_OPTIMISTIC_H":estimate["optimistic_hours"],"ESTIMATED_1B_STAGE1_WALL_TIME_CENTRAL_H":estimate["central_hours"],"ESTIMATED_1B_STAGE1_WALL_TIME_CONSERVATIVE_H":estimate["conservative_hours"],
"ACTUAL_TRAIN_SPLIT_CENTRAL_H":estimate["actual_train_split_central_hours"],
"GPU_METRICS":t["gpu"],"THROUGHPUT_MEASUREMENT":t,
"CHECKPOINT_ESTIMATED_SIZE":policy["measured_debug_dcp_bytes"],"CHECKPOINT_CADENCE":"step25 then every250 updates, plus final19168","CHECKPOINT_RETENTION_POLICY":str(R/"config/CHECKPOINT_RETENTION_POLICY.json"),
"FORMAL_CONFIG_PATH":str(cfgpath),"FORMAL_CONFIG_SHA256":sha(cfgpath),
"FORMAL_LAUNCH_SCRIPT":str(R/"bin/launch_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh"),"FORMAL_STATUS_SCRIPT":str(R/"bin/status_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh"),
"FORMAL_DATA_ORDER_PREFLIGHT":"PASS","FORMAL_PRELAUNCH_AUDIT":"PASS","FORMAL_STAGE1_TRAINING_STARTED":False,
"warnings":manifest["warnings"],"historical_failed_runs_preserved":True,"deterministic_runtime":cfg["deterministic_runtime"],
"file_sha256":files,"requirement_evidence":manifest["requirement_evidence"],"independent_auditor":review,"created_at_unix":time.time()
}
p=R/"provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V1.json"
assert not p.exists(),"FINAL_AUTHORITY_ALREADY_EXISTS"
p.write_text(json.dumps(authority,indent=2)+"\n")
p.with_suffix(".sha256").write_text(sha(p)+"  "+p.name+"\n")
print(json.dumps({"authority":str(p),"sha256":sha(p),"status":authority["QWEN3_4B_STAGE1_PRELAUNCH_STATUS"],"formal_training_started":False}))
