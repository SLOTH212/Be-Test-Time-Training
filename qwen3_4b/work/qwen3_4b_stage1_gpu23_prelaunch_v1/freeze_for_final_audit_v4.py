import os,pathlib,json,hashlib,yaml,math,shutil,time,subprocess,sys,importlib.metadata
R=pathlib.Path("/path/to/ttt");W=R/"work/qwen3_4b_stage1_gpu23_prelaunch_v1"
A=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_determinism_v4"
A1=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_v1"
A2=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_corrected_v2"
A3=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_final_v3"
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+"\n")
for name in ["SUITE_COMPLETED.json","STEADY_STATE_THROUGHPUT.json","REAL_RUNTIME_GATES.json","REAL_DCP_AND_RESUME_RNG.json","REAL_RESUME_NUMERIC_PARITY.json","REAL_FAILURE_RECOVERY_NUMERIC_PARITY.json","FORMAL_DETERMINISTIC_CONTROLLER_REGRESSION.json"]:
 assert load(A/name)["status"]=="PASS",name
assert not (R/"runs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1").exists()
assert not (R/"checkpoints/qwen3_4b_stage1_1b_32k_gpu23_formal_v1_model_only").exists()
# Recheck current environment without importing CUDA.
env=load(A3/"FULL_ENVIRONMENT_IDENTITY.json")
assert env["python"]==sys.version and env["executable"]==str(pathlib.Path(sys.executable).resolve())
for name,version in env["packages"].items():assert importlib.metadata.version(name)==version
source=load(A3/"RUNTIME_SOURCE_SHA256.json")
for rel,h in source.items():assert sha(R/"src/training_runtime"/rel)==h,rel
for p,h in load(A3/"MODEL_METADATA_IDENTITY.json")["metadata_file_sha256"].items():assert sha(p)==h,p
baseline=load(A2/"OUTSIDE_CACHE_BASELINE.json")
now={}
for rel in baseline:
 p=pathlib.Path("/home/USER")/rel
 now[rel]={str(f):{"mtime_ns":f.stat().st_mtime_ns,"bytes":f.stat().st_size} for f in p.rglob("*") if f.is_file()} if p.exists() else {}
assert now==baseline,"OUTSIDE_CACHE_CHANGED"
for key in ["TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","TORCH_EXTENSIONS_DIR","CUDA_CACHE_PATH","TMPDIR"]:
 assert pathlib.Path(os.environ[key]).resolve().is_relative_to(R)
write(A/"OUTPUT_AND_CACHE_POLICY_AUDIT.json",{"status":"PASS_SINCE_REMEDIATION","checked_at_unix":time.time(),"baseline_sha256":sha(A2/"OUTSIDE_CACHE_BASELINE.json"),"outside_cache_file_inventory_sizes_mtimes_unchanged":True,"outside_cache_content_hash_comparison":"Not available in original baseline; no byte-identical-content claim","historical_violation_preserved":True,"historical_incident":str(A1/"CACHE_POLICY_INCIDENT_AND_STOP.json"),"all_current_configured_cache_paths_under_authorized_root":True,"historical_outside_files_removed":False})
checkpoint_actual=sha(R/"src/training_runtime/code/lib/distributed_checkpoint.py")
for receipt in ["RETENTION_PUBLICATION_FAULT_TEST.json","REAL_RETENTION_ROUNDTRIP.json"]:
 d=load(A3/receipt)
 assert d["status"]=="PASS" and d["checkpoint_module_sha256"]==checkpoint_actual,receipt
for receipt in [A1/"MODEL_IDENTITY.json",A3/"MODEL_METADATA_IDENTITY.json",A1/"FROZEN_INPUT_HASH_VERIFICATION.json",A1/"REAL_DATALOADER_PREFLIGHT.json",A2/"REAL_SCHEMA_AND_METADATA_PREFLIGHT.json",R/"audits/BATCH_SEMANTICS_AUDIT.json",R/"audits/STAGE1_STOP_SEMANTICS_AUDIT.json"]:
 assert load(receipt)["status"]=="PASS",str(receipt)
candidate=W/"qwen3_4b_stage1_1b_32k_gpu23_formal_CANDIDATE_NOT_FROZEN.yaml"
candidate_cfg=yaml.safe_load(candidate.read_text())
debug_cfg=yaml.safe_load((W/"determinism_probe.yaml").read_text())
assert candidate_cfg["input_identity"]==debug_cfg["input_identity"]
assert candidate_cfg["deterministic_runtime"]==debug_cfg["deterministic_runtime"]
for key in ["world_size","physical_gpu_allowlist","context_length","ttt_chunk_size","ttt_layers","micro_batch_size","gradient_accumulation","global_batch_size","precision","gradient_checkpointing","seed","ttt_lr","optimizer","outer_lr","betas","eps","weight_decay","outer_grad_clip","scheduler","model_base","train_data","validation_data"]:
 assert candidate_cfg[key]==debug_cfg[key],key
expected_paths={"worker_sha256":R/"src/training_runtime/code/workers/distributed_train_worker.py","checkpoint_module_sha256":R/"src/training_runtime/code/lib/distributed_checkpoint.py","dataset_authority_sha256":R/"datasets/authority/DATASET_AUTHORITY_V3.json","data_manifest_sha256":R/"datasets/manifests/FINAL_DATA_FILE_HASHES.jsonl","model_identity_sha256":A1/"MODEL_IDENTITY.json","cache_policy_sha256":R/"configs/paths.env"}
for key,p in expected_paths.items():assert candidate_cfg["input_identity"][key]==sha(p),key
assert sha(candidate_cfg["execution_wrapper"])==candidate_cfg["deterministic_runtime"]["wrapper_sha256"]==sha(W/"deterministic_stage1_wrapper_probe.py")
t=load(A/"STEADY_STATE_THROUGHPUT.json");cp=R/"config/CHECKPOINT_RETENTION_POLICY.json";policy=load(cp)
size=t["debug_checkpoint_bytes"]
policy.update(status="FROZEN",measured_debug_dcp_bytes=size,estimated_resident_dcp_bytes=2*size,estimated_transient_peak_dcp_bytes=3*size,storage_reserve_bytes_recommended=math.ceil(1.25*(3*size+10_000_000_000)),measured_checkpoint_seconds=t["wall_time_estimate"]["overhead"]["checkpoint_samples_seconds"],measured_throughput_receipt=str(A/"STEADY_STATE_THROUGHPUT.json"))
free=shutil.disk_usage(R).free
assert free>policy["storage_reserve_bytes_recommended"]
policy["available_disk_bytes_at_freeze"]=free;write(cp,policy)
candidate=W/"qwen3_4b_stage1_1b_32k_gpu23_formal_CANDIDATE_NOT_FROZEN.yaml"
cfg=yaml.safe_load(candidate.read_text())
cfg.update(config_class="FORMAL_STAGE1_V3_GPU23_FROZEN_V1",capacity_parameters_frozen=True,formal_gpu_policy_frozen=True,formal_batch_semantics_frozen=True)
cfg["minimum_free_memory_mib"]={i:max(43000,math.ceil(t["gpu"][str(i)]["torch_peak_reserved_bytes"]/2**20)+4096) for i in [2,3]}
assert max(cfg["minimum_free_memory_mib"].values())<49000
cfg["throughput_receipt"]=str(A/"STEADY_STATE_THROUGHPUT.json")
cfg["real_runtime_gate_receipt"]=str(A/"REAL_RUNTIME_GATES.json")
target=R/"configs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1.yaml"
assert not target.exists(),"FROZEN_CONFIG_ALREADY_EXISTS"
target.write_text(yaml.safe_dump(cfg,sort_keys=False))
order=load(A2/"FORMAL_DATA_ORDER_PREFLIGHT.json")
order.update(formal_config=str(target),formal_config_sha256=sha(target),current_real_trace_receipt=str(A/"STEADY_STATE_THROUGHPUT.json"))
write(A/"FORMAL_DATA_ORDER_PREFLIGHT.json",order)
contract=load(R/"provenance/STAGE1_SCIENTIFIC_CONTRACT.json")
contract["independent_source_review"]="Frozen core and two-rank batch semantics reviewed; deterministic wrapper reviewed as engineering-only; current runtime gates independently indexed for final audit."
contract["deterministic_engineering_controls"]={"wrapper":cfg["execution_wrapper"],"wrapper_sha256":cfg["deterministic_runtime"]["wrapper_sha256"],"scientific_semantics_unchanged":True,"current_runtime_evidence":str(A/"REAL_RUNTIME_GATES.json")}
write(R/"provenance/STAGE1_SCIENTIFIC_CONTRACT.json",contract)
# Explicit evidence index; independent reviewer supplies the final adjudication.
evidence={
"0_hard_safety":[A/"OUTPUT_AND_CACHE_POLICY_AUDIT.json",A1/"GPU_SHARING_AUTHORIZATION.json",A1/"NCCL_GPU_MAPPING.json"],
"1_shared_guard":[R/"bin/check_shared_assets.sh",R/"provenance/QWEN3_4B_SHARED_MODEL_GUARD_SHA256SUMS.txt"],
"2_environment":[A3/"FULL_ENVIRONMENT_IDENTITY.json"],
"3_runtime":[A3/"RUNTIME_SOURCE_SHA256.json",A3/"LEVEL0.log",A3/"LEVEL1.log",R/"src/training_runtime/provenance/MULTIGPU_RUNTIME_AUTHORITY_V2.json"],
"4_science":[R/"provenance/STAGE1_SCIENTIFIC_CONTRACT.json"],
"5_model":[A1/"MODEL_IDENTITY.json",A3/"MODEL_METADATA_IDENTITY.json"],
"6_7_dataset":[A1/"FROZEN_INPUT_HASH_VERIFICATION.json",R/"datasets/authority/DATASET_AUTHORITY_V3.json",R/"datasets/manifests/FINAL_DATA_FILE_HASHES.jsonl"],
"8_loader":[A1/"REAL_DATALOADER_PREFLIGHT.json",A2/"REAL_SCHEMA_AND_METADATA_PREFLIGHT.json"],
"9_10_batch":[R/"audits/BATCH_SEMANTICS_AUDIT.json"],
"11_12_gpu":[A1/"NCCL_GPU_MAPPING.json",A1/"GPU_SHARING_AUTHORIZATION.json"],
"13_17_real_runtime":[A/"REAL_RUNTIME_GATES.json",A/"STEADY_STATE_THROUGHPUT.json",A1/"REAL_METADATA_CORE_REGRESSION.json",A1/"FULL_CANONICAL_RECORD_ACCOUNTING.json"],
"18_20_dcp_resume_failure":[A/"REAL_DCP_AND_RESUME_RNG.json",A/"REAL_RESUME_NUMERIC_PARITY.json",A/"REAL_FAILURE_RECOVERY_NUMERIC_PARITY.json",A/"FAILURE_C_INJECT_EXIT.json",A/"FAILURE_C_INJECT.log"],
"21_23_measurement":[A/"STEADY_STATE_THROUGHPUT.json",A/"GPU23_UTILIZATION_IO.jsonl",A/"PERFORMANCE_REGRESSION_COMPARISON.json"],
"24_validation":[A/"REAL_RUNTIME_GATES.json"],
"25_storage":[cp,A3/"RETENTION_PUBLICATION_FAULT_TEST.json",A3/"REAL_RETENTION_ROUNDTRIP.json"],
"26_walltime":[A/"STEADY_STATE_THROUGHPUT.json",R/"audits/STAGE1_STOP_SEMANTICS_AUDIT.json"],
"27_29_freeze":[target],
"30_order":[A/"FORMAL_DATA_ORDER_PREFLIGHT.json"],
"31_32_scripts":[R/"bin/launch_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh",R/"bin/status_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh",R/"bin/qwen3_4b_stage1_gpu23_formal_control_v1.py",pathlib.Path(cfg["execution_wrapper"]),A/"FORMAL_DETERMINISTIC_CONTROLLER_REGRESSION.json",A2/"FORMAL_CONTROLLER_REGRESSION.json",A2/"DETACHED_PROCESS_PATTERN_AUDIT.json"],
"34_no_formal":[A/"SUITE_COMPLETED.json"]
}
files={str(p):sha(p) for group in evidence.values() for p in group}
extra=[R/"bin/activate_ttt.sh",R/"configs/paths.env",W/"determinism_probe.yaml",W/"real_edge_deterministic_v4.yaml",W/"deterministic_stage1_wrapper_probe.py",W/"run_deterministic_completion_v4.py",W/"final_real_evidence_v4.py",W/"analyze_deterministic_throughput_v4.py",A3/"PARITY_ACCEPTANCE_CRITERIA.json",A3/"REAL_RESUME_NUMERIC_PARITY.json",A1/"CACHE_POLICY_INCIDENT_AND_STOP.json",A1/"CACHE_POLICY_REMEDIATION.json"]
extra+=list((R/"src/training_runtime/provenance").glob("*"))
extra+=list((R/"datasets/authority").glob("*"))
for p in extra:
 if p.is_file():files[str(p)]=sha(p)
for rel,h in source.items():files[str(R/"src/training_runtime"/rel)]=h
files.update(load(A3/"MODEL_METADATA_IDENTITY.json")["metadata_file_sha256"])
for folder in ["throughput_A","fresh_B","failure_C","edge_one_token","edge_odd_final"]:
 p=R/"runs/qwen3_4b_gpu23_prelaunch_determinism_v4"/folder
 for name in ["training.jsonl","complete.json","validation_smoke.json"]:files[str(p/name)]=sha(p/name)
subprocess.run(["bash","-n",str(R/"bin/launch_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh")],check=True)
subprocess.run(["bash","-n",str(R/"bin/status_qwen3_4b_stage1_1b_32k_gpu23_formal_v1.sh")],check=True)
for path in [R/"bin/qwen3_4b_stage1_gpu23_formal_control_v1.py",pathlib.Path(cfg["execution_wrapper"])]:compile(path.read_text(),str(path),"exec")
write(A/"PRELAUNCH_AUDIT_INPUT_MANIFEST.json",{"status":"READY_FOR_INDEPENDENT_REVIEW","created_at_unix":time.time(),"formal_config_sha256":sha(target),"file_sha256":files,"requirement_evidence":{k:list(map(str,v)) for k,v in evidence.items()},"pending":["33 final independent prelaunch adjudication","35 final authority and checksum","36 complete requirement audit"],"formal_training_started":False,"warnings":["Historical outside-root Triton cache write preserved; current runs confined after remediation","Existing world-writable shared model files; inputs unmodified","GPU utilization counter reports zero during real training; raw telemetry preserved","Historical 1.7B implemented batch verified, completed-run log not independently verified","Nominal 1B dataset allocation is 990M train plus 10M validation; actual train input989996971","Throughput sample is first50 canonical records and shared GPUs; corpus-wide extrapolation uncertain","Strict deterministic candidate measured 2.85x slower than prior non-deterministic settings; individual operator cause and performance optimum not profiled"]})
print(json.dumps({"status":"READY_FOR_INDEPENDENT_REVIEW","formal_config_sha256":sha(target),"minimum_free_memory_mib":cfg["minimum_free_memory_mib"],"file_hash_count":len(files),"formal_training_started":False}))
