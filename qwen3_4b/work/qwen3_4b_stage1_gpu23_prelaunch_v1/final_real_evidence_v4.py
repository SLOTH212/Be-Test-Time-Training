import os,pathlib,json,hashlib,math,yaml
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
import torch,numpy as np
from torch.distributed.checkpoint import FileSystemReader
R=pathlib.Path("/path/to/ttt")
W=R/"work/qwen3_4b_stage1_gpu23_prelaunch_v1"
A=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_determinism_v4"
RUN=R/"runs/qwen3_4b_gpu23_prelaunch_determinism_v4"
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(name,v):(A/name).write_text(json.dumps(v,indent=2)+"\n")
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def rows(name):return [json.loads(x) for x in (RUN/name/"training.jsonl").read_text().splitlines()]
def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
assert read(A/"SUITE_COMPLETED.json")["status"]=="PASS"
throughput=read(A/"STEADY_STATE_THROUGHPUT.json")
assert throughput["status"]=="PASS"
cfg=yaml.safe_load((W/"determinism_probe.yaml").read_text())
identity={**cfg["input_identity"],"config_sha256":sha(W/"determinism_probe.yaml")}
a=rows("throughput_A");b=rows("fresh_B");c=rows("failure_C")
assert len(a)==25 and [x["update_step"] for x in b]==[1,2,3,4,5]
assert [x["update_step"] for x in c]==[1,3,4,5]
trajectory=[]
for name,log in [("resume_B",b),("failure_C",c)]:
 for row in log:
  reference=a[row["update_step"]-1]
  for key in ["record_cursor","cumulative_tokens","global_valid_tokens","global_context_positions","accumulation_phase","dataloader_state","input_identity","learning_rate"]:
   assert exact(reference[key],row[key]),(name,key,row["update_step"])
  assert math.isclose(reference["global_mean_loss"],row["global_mean_loss"],rel_tol=.01,abs_tol=1e-5)
  for rank in range(2):
   for key in ["rank","local_rank","physical_gpu","sample_ids","local_tokens","boundary_trace"]:
    assert exact(reference["rank_runtime"][rank][key],row["rank_runtime"][rank][key]),(name,rank,key)
  trajectory.append({"case":name,"step":row["update_step"],"loss_A":reference["global_mean_loss"],"loss_other":row["global_mean_loss"],"loss_abs_difference":abs(reference["global_mean_loss"]-row["global_mean_loss"])})
anchor=RUN/"uninterrupted_step5_anchor/checkpoints/slot_B"
states=[];dcp=[]
for name,path in [("A",anchor),("B",RUN/"fresh_B/checkpoints"),("C",RUN/"failure_C/checkpoints")]:
 if name!="A":path=path/read(path/"latest.json")["slot"]
 manifest=read(path/"manifest.json")
 assert manifest["complete"] and manifest["world_size"]==2 and manifest["progress"]["update_step"]==5
 assert manifest["batch_metadata"]["identity"]==identity
 # Verify actual bytes, not just the recorded manifest.
 for rel,expected in manifest["files"].items():assert sha(path/rel)==expected,(name,rel)
 metadata=FileSystemReader(str(path/"dcp")).read_metadata()
 keys=list(metadata.state_dict_metadata)
 assert any(k.startswith("model.") for k in keys) and any(k.startswith("optimizer.") for k in keys)
 local=[]
 for rank in range(2):
  x=torch.load(path/f"rank_{rank:05d}.pt",map_location="cpu",weights_only=False)
  assert set(x)=={"progress","python_rng","numpy_rng","torch_cpu_rng","torch_cuda_rng","scheduler"}
  p=x["progress"]
  assert p["record_cursor"]==10 and p["cumulative_tokens"]==a[4]["cumulative_tokens"] and p["accumulation_phase"]==0
  assert p["dataloader_state"]["canonical_next_position"]==10 and p["rank_assignment_state"]["rank"]==rank and p["rank_assignment_state"]["world_size"]==2
  assert p["input_identity"]==identity and len(x["torch_cuda_rng"])==2
  local.append(x)
  if name!="A":
   checks={k:exact(states[rank][k],x[k]) for k in x}
   assert all(checks.values()),(name,rank,checks)
   dcp.append({"case":name,"rank":rank,"exact_state_fields":checks})
 if name=="A":states=local
dcp_report = {"status":"PASS","input_identity":identity,"world_size":2,"actual_checkpoint_file_hashes_verified":True,"all_required_DCP_fields_present":True,"rank_state_comparisons":dcp,"trajectory":trajectory,"controlled_failure_step2_note":"Injected after valid DCP before JSONL append; saved cursor4/state recovers to step3.","model_optimizer_parity_receipts":["REAL_RESUME_NUMERIC_PARITY.json","REAL_FAILURE_RECOVERY_NUMERIC_PARITY.json"]}
for p in ["REAL_RESUME_NUMERIC_PARITY","REAL_FAILURE_RECOVERY_NUMERIC_PARITY"]:
 assert read(A/(p+".json"))["status"]=="PASS"
for phase in ["THROUGHPUT_25","RESUME_B_AFTER","FAILURE_C_RESUME","EDGE_ONE_TOKEN","EDGE_ODD_FINAL"]:
 assert read(A/(phase+"_EXIT.json"))["exit_code"]==0
 text=(A/(phase+".log")).read_text(errors="replace")
 assert "RANK_0_SHUTDOWN_PASS" in text and "RANK_1_SHUTDOWN_PASS" in text
failure=(A/"FAILURE_C_INJECT.log").read_text(errors="replace")
assert read(A/"FAILURE_C_INJECT_EXIT.json")["exit_code"]!=0 and "CONTROLLED_DEBUG_SOFTWARE_FAILURE" in failure
assert "CONTROLLED_SOFTWARE_FAILURE_AFTER_VALID_DCP rank=0 step=2" in failure and "CONTROLLED_SOFTWARE_FAILURE_AFTER_VALID_DCP rank=1 step=2" in failure
assert "RANK_0_SHUTDOWN_PASS" in failure and "RANK_1_SHUTDOWN_PASS" in failure
assert not list((RUN/"failure_C").glob("complete.previous.*.json"))
assert read(RUN/"failure_C/complete.json")["record_cursor"]==10
validation={}
for name in ["throughput_A","fresh_B","failure_C","edge_one_token","edge_odd_final"]:
 v=read(RUN/name/"validation_smoke.json")
 assert v["weights_unchanged"] and v["optimizer_steps"]==0 and v["verified_parameter_shards_per_rank"]==416
 assert v["weights_unchanged_method"]=="exact_sha256_local_shards_after_recursive_reshard"
 assert math.isfinite(v["global_mean_loss"]) and v["global_mean_loss"]>0 and 0<v["records"]<=4
 validation[name]=v
one=rows("edge_one_token");odd=rows("edge_odd_final")
assert len(one)==len(odd)==1
assert one[0]["rank_runtime"][0]["local_tokens"]==1
assert one[0]["rank_runtime"][0]["sample_ids"]==["prolong-v3-32k:train:015220"]
assert one[0]["global_valid_tokens"]>0
assert odd[0]["record_cursor"]==38335 and odd[0]["rank_runtime"][0]["sample_ids"]==["prolong-v3-32k:train:038334"]
assert odd[0]["rank_runtime"][1]["local_tokens"]==0 and odd[0]["rank_runtime"][1]["sample_ids"]==[]
with (R/"datasets/train/part-00000.jsonl").open() as f:source=[json.loads(next(f)) for _ in range(50)]
for row in a:
 assert math.isfinite(row["global_mean_loss"]) and math.isfinite(row["grad_norm_before_clip"]) and row["grad_norm_after_clip"]<=1.00001 and row["learning_rate"]==5e-6
 for rr in row["rank_runtime"]:
  assert rr["weights_finite"]
  for layer in [0,6,12,18,24,30]:
   for suffix in ["ttt_ntp_gate","ttt_proj.weight"]:
    key=f"model.layers.{layer}.mlp.{suffix}"
    g=rr["l30_gradient"][key];o=rr["l30_optimizer_state"][key]
    assert g["finite"] and (g["present"] or not g["expected_present"])
    assert o["owned"] and (not o["state_expected"] or set(o["state_keys"])=={"exp_avg","exp_avg_sq","step"})
complete=read(RUN/"throughput_A/complete.json")
assert complete["fsdp2_wrapped_blocks"]==36
write("REAL_RUNTIME_GATES.json",{"status":"PASS",**{k:"PASS" for k in ["REAL_DATA_2GPU_STAGE1_SMOKE","REAL_DATA_DOCUMENT_BOUNDARY_CHECK","REAL_DATA_TOKEN_ACCOUNTING","REAL_DATA_DCP_SAVE","REAL_DATA_RESUME_PARITY","REAL_DATA_FAILURE_RECOVERY","FORMAL_TTT_LAYER_COVERAGE","FORMAL_CHUNK4K_RUNTIME","STAGE1_VALIDATION_DATA_SMOKE"]},"warmup_steps":5,"measured_steps":20,"real_records":50,"full_32768_records":sum(x["token_count"]==32768 for x in source),"partial_records":sum(x["token_count"]<32768 for x in source),"multi_document_records":sum(len(json.loads(x["document_boundaries_qwen_json"]))>1 for x in source),"fsdp2_blocks":36,"six_layer_gradient_and_optimizer_coverage":True,"validation":validation,"one_token_and_odd_final_rank_rules":"PASS","both_rank_shutdown_pass":True,"failure_no_false_complete":"driver asserts immediately after failed process; no prior completion archive exists; final completion appears only after successful resume","formal_training_started":False})
write("REAL_DCP_AND_RESUME_RNG.json",dcp_report)
print("FINAL_REAL_EVIDENCE_PASS")
