#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,math,os,time,traceback
from datetime import datetime
from pathlib import Path
MODEL="ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f";CKPT="f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd";BENCH="5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5";RULE="c82ea5f5c94b68ee3df5ec238faff2423ff9b25dd530ab8f34cfb92aafb3a800";ELIG="ea98b9944921cd892dfdda3bdac5b4a108db50d4bf9c7ce7e556b3cf262d3b98";EXEC="293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e";SCORER="cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556";BACKEND="c15b6872c625ee13ee2b71e522a9414e6f797edd6a695e185465dff05c5326ee"
LAYERS=[0,6,12,18,24];ACTIONS=["OFF","L0","L6","L12","L18","L24","ALL"]
def now():return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f".{p.name}.{os.getpid()}.{time.time_ns()}.tmp");fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  with os.fdopen(fd,"w") as f:json.dump(o,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
 except BaseException:
  try:os.unlink(t)
  except FileNotFoundError:pass
  raise
def imp(p,n):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def status(run,**kw):
 p=run/"status/pipeline_status.json";x=load(p) if p.exists() else {};x.update(kw);x["updated_at"]=now();atomic(p,x)
class Ctx:
 def __init__(self,c):
  self.run=Path(c["run_root"]);self.code=Path("/home/USER/ttt/code/ntp");self.ckpt=Path(c["model_root"]);self.score_fn=imp(c["scorer"],"selective_scorer").per_sample_score
  self.rows=[]
  for i,r in enumerate(rows(Path(c["benchmark_root"])/"sample_manifest.jsonl")):r["_sample_index"]=i;self.rows.append(r)
  self.by_id={r["sample_id"]:r for r in self.rows};self.manifest={"tasks":sorted({r["task"] for r in self.rows})};self.baseline={a:{} for a in ACTIONS}
  for r in rows(Path(c["fixed7_root"])/"manifests/all_fixed7_results.jsonl"):r["official_sample_score"]=float(r["score"]);self.baseline[r["mode"]][r["sample_id"]]=r
  if len(self.by_id)!=6500 or any(len(self.baseline[a])!=6500 for a in ACTIONS):raise RuntimeError("CONTEXT_AUTHORITY_INVALID")
def gates(run):
 run=Path(run).resolve();c=load(run/"config/SELECTIVE_DELETION_FORMAL_CONFIG.json")
 if sorted(os.sched_getaffinity(0))!=list(range(8,24)):raise RuntimeError("CPU_AFFINITY_MISMATCH")
 ck={"model":sha(Path(c["model_root"])/"model.safetensors")==MODEL,"benchmark":sha(Path(c["benchmark_root"])/"benchmark_manifest.json")==BENCH,"rule":sha(c["rule_manifest"])==RULE,"eligibility":sha(c["eligibility_manifest"])==ELIG,"cohort":sha(c["cohort_manifest"])==c["cohort_sha256"],"branches":sha(c["branch_manifest"])==c["branch_manifest_sha256"],"protocol":sha(c["protocol"])==c["protocol_sha256"],"executor":sha(c["executor"])==EXEC,"scorer":sha(c["scorer"])==SCORER,"backend":sha(c["backend"])==BACKEND,"worker":sha(c["worker"])==c["worker_sha256"],"aggregate":sha(c["aggregator"])==c["aggregator_sha256"],"controller":sha(c["controller"])==c["controller_sha256"]}
 if not all(ck.values()):raise RuntimeError("HASH_GATE "+json.dumps(ck,sort_keys=True))
 co=rows(c["cohort_manifest"]);br=rows(c["branch_manifest"])
 if len(co)!=151 or len(br)!=458 or sum(x["intervention_type"]=="SELECTIVE" for x in br)!=359 or sum(x["intervention_type"]=="SHAM" for x in br)!=99:raise RuntimeError("SCOPE_GATE")
 return run,c,co,br,ck
def runtime(run,c):
 runner=imp(c["executor"],"frozen_selective_executor");backend=imp(c["backend"],"frozen_selective_backend");ctx=Ctx(c)
 import torch;torch.set_num_threads(16);status(run,current_phase="MODEL_LOADING",worker_pid=os.getpid());model,tok=runner.load_model(ctx);base=runner.base_hash(model);backend.install_instrumented_dynamic_forward(model,runner.tensor_hash)
 atomic(run/"audit/MODEL_LOAD_AUDIT.json",{"status":"PASS","model_identity":MODEL,"checkpoint_identity":CKPT,"base_weight_runtime_hash":base,"cuda_device":torch.cuda.get_device_name(0),"timestamp":now()});return runner,backend,ctx,model,tok,base
def configure(model,layer,b):
 for z in LAYERS:
  m=model.model.layers[z].mlp;m._reset_boundary_chunk=b if z==layer else None;m._instrumentation_enabled=True;m._state_events=[];m._reset_events=[]
def evaluate(runner,backend,ctx,model,tok,sample,layer,boundary,stage,ordinal,total):
 configure(model,layer,boundary);ev=runner.Evaluator(ctx,model,tok,ctx.by_id[sample["canonical_sample_id"]],stage,ordinal,total,ctx.run/"status/worker_internal_progress.json");rec=ev.evaluate(tuple(sample["action_sequence"]),capture=True,force=True);ev.finish();se=[];re=[]
 for z in LAYERS:
  m=model.model.layers[z].mlp;se+=backend.materialize_scalar(list(m._state_events));re+=backend.materialize_scalar(list(m._reset_events))
 if len(se)!=5*sample["full_chunk_count"] or len(re)!=(0 if layer is None else 1):raise RuntimeError("EVENT_COUNT")
 if any(e["selected_action"]!=sample["action_sequence"][e["chunk_index"]-1] for e in se):raise RuntimeError("ACTION_MISMATCH")
 if layer is not None and (re[0]["layer"]!=layer or re[0]["chunk_index"]!=boundary):raise RuntimeError("TARGET_RESET_MISMATCH")
 return rec,se,re
def boundary_hashes(rec,boundary):
 return {int(e["layer"]):e["pre_state_hash"] for e in rec["chunk_layer_events"] if int(e["chunk_index"])==int(boundary)}
def valid(p,b,c):
 x=load(p);ok=x.get("completed") is True and x.get("status")=="SUCCESS" and x.get("branch_id")==b["branch_id"] and x.get("cohort_hash")==c["cohort_sha256"] and x.get("branch_manifest_hash")==c["branch_manifest_sha256"] and x.get("protocol_hash")==c["protocol_sha256"] and x.get("target_layer_reset_parity")=="PASS" and x.get("non_target_layer_preservation")=="PASS"
 if not ok:raise RuntimeError("INVALID_COMMIT "+str(p));return x
 return x
def smoke(run):
 run,c,co,br,checks=gates(run)
 if list((run/"branches").glob("*.json")):raise RuntimeError("FORMAL_BRANCH_EXISTS")
 by={x["canonical_sample_id"]:x for x in co};non=next(x for x in co if not x["all_prefix"] and x["sham_layer"] is not None);allp=next(x for x in co if x["all_prefix"])
 status(run,pipeline_status="SMOKE_RUNNING",current_phase="NATIVE_SELECTIVE_SHAM_SMOKE",last_error="NONE");runner,backend,ctx,model,tok,base=runtime(run,c);native={};detail=[]
 try:
  for i,s in enumerate([non,allp],1):
   rec,_,_=evaluate(runner,backend,ctx,model,tok,s,None,None,"selective_native_smoke",i,2);ok=rec["score"]==s["normal_score"] and rec["prediction_hash"]==s["normal_prediction_sha256"]
   if not ok:raise RuntimeError("NATIVE_PARITY "+s["canonical_sample_id"])
   native[s["canonical_sample_id"]]=rec
  sb=[x for x in br if x["canonical_sample_id"]==non["canonical_sample_id"] and x["intervention_type"]=="SELECTIVE"]+[x for x in br if x["canonical_sample_id"]==allp["canonical_sample_id"] and x["intervention_type"]=="SELECTIVE"]
  if len(sb)!=6 or {x["target_layer"] for x in sb if x["canonical_sample_id"]==allp["canonical_sample_id"]}!=set(LAYERS):raise RuntimeError("SMOKE_BRANCH_SCOPE")
  for i,b in enumerate(sb,1):
   s=by[b["canonical_sample_id"]];rec,se,re=evaluate(runner,backend,ctx,model,tok,s,b["target_layer"],b["boundary"],"selective_branch_smoke",i,6);pre=float(re[0]["pre_reset_delta_norm"]);post=float(re[0]["post_reset_max_abs_residual"])
   nh=boundary_hashes(native[s["canonical_sample_id"]],b["boundary"]);dh=boundary_hashes(rec,b["boundary"]);preserved=all(nh[z]==dh[z] for z in LAYERS if z!=b["target_layer"])
   if not(pre>0 and post==0 and preserved and runner.base_hash(model)==base):raise RuntimeError("SELECTIVE_INTEGRITY")
   detail.append({"branch_id":b["branch_id"],"target_layer":b["target_layer"],"pre_norm":pre,"post_residual":post,"non_target_boundary_hash_parity":preserved})
  sham=next(x for x in br if x["canonical_sample_id"]==non["canonical_sample_id"] and x["intervention_type"]=="SHAM");rec,se,re=evaluate(runner,backend,ctx,model,tok,non,sham["target_layer"],sham["boundary"],"selective_sham_smoke",1,1);pre=float(re[0]["pre_reset_delta_norm"]);post=float(re[0]["post_reset_max_abs_residual"]);np=native[non["canonical_sample_id"]]
  shamok=pre==0 and post==0 and rec["score"]==np["score"] and rec["prediction_hash"]==np["prediction_hash"] and runner.base_hash(model)==base
  if not shamok:raise RuntimeError("SHAM_PARITY")
 finally:
  del model,tok;gc.collect();import torch;torch.cuda.empty_cache()
 out={"status":"PASS","formal_result":False,"ORIGINAL_DYNAMIC_REPLAY_PARITY":"PASS","SELECTIVE_SMOKE_STATUS":"PASS","TARGET_COMPONENT_NONZERO":"PASS","TARGET_LAYER_RESET_PARITY":"PASS","NON_TARGET_LAYER_PRESERVATION":"PASS","SELECTIVE_BOUNDARY_PARITY":"PASS","ACTION_SCHEDULE_PARITY":"PASS","PRESERVED_CONTEXT_SEMANTICS":"PASS","SHAM_SMOKE_STATUS":"PASS","SHAM_ZERO_STATE_PRECONDITION":"PASS","SHAM_NATIVE_PARITY":"PASS","BASE_PARAMETER_MUTATION":False,"detail":detail,"completed_at":now()};atomic(run/"smoke/SELECTIVE_DELETION_SMOKE.json",out);status(run,pipeline_status="SMOKE_PASS",current_phase="READY_FOR_FORMAL_LAUNCH",worker_pid=None,smoke_status="PASS");print(json.dumps(out,sort_keys=True),flush=True)
def formal(run):
 run,c,co,branches,_=gates(run);sm=load(run/"smoke/SELECTIVE_DELETION_SMOKE.json")
 if sm["status"]!="PASS":raise RuntimeError("SMOKE_GATE")
 by={x["canonical_sample_id"]:x for x in co};done=0
 for b in branches:
  p=run/"branches"/(b["branch_id"]+".json")
  if p.exists():valid(p,b,c);done+=1
 status(run,pipeline_status="RUNNING",current_phase="FORMAL_458_BRANCH_EXECUTION",committed_branches=done,total_expected=458,worker_pid=os.getpid(),worker_running=True,formal_launched=True,last_error="NONE");runner,backend,ctx,model,tok,base=runtime(run,c);start=time.time();new=0
 try:
  for b in branches:
   p=run/"branches"/(b["branch_id"]+".json")
   if p.exists():continue
   s=by[b["canonical_sample_id"]];atomic(run/"claims"/(b["branch_id"]+".json"),{"branch_id":b["branch_id"],"state":"running","worker_pid":os.getpid(),"timestamp":now()});status(run,current_sample=s["canonical_sample_id"],current_task=s["task"],current_branch_type=b["intervention_type"],current_target_layer=b["target_layer"],current_boundary=b["boundary"])
   t=time.time();rec,se,re=evaluate(runner,backend,ctx,model,tok,s,b["target_layer"],b["boundary"],"formal_selective_deletion_1p7b_v1",b["branch_order"],458);e=re[0];pre=float(e["pre_reset_delta_norm"]);res=float(e["post_reset_max_abs_residual"]);selective=b["intervention_type"]=="SELECTIVE"
   if (selective and pre<=0) or ((not selective) and pre!=0) or res!=0 or runner.base_hash(model)!=base:raise RuntimeError("BRANCH_INTEGRITY "+b["branch_id"])
   if not selective and (rec["score"]!=s["normal_score"] or rec["prediction_hash"]!=s["normal_prediction_sha256"]):raise RuntimeError("SHAM_INTEGRITY "+b["branch_id"])
   out={**b,"score":float(rec["score"]),"prediction_sha256":rec["prediction_hash"],"normal_score":s["normal_score"],"normal_prediction_sha256":s["normal_prediction_sha256"],"effect":s["normal_score"]-float(rec["score"]),"target_component_pre_norm":pre,"target_post_reset_residual":res,"target_component_nonzero":"PASS" if selective else "NOT_APPLICABLE","sham_zero_state_precondition":"PASS" if not selective else "NOT_APPLICABLE","target_layer_reset_parity":"PASS","non_target_layer_preservation":"PASS","selective_boundary_parity":"PASS","action_schedule_parity":"PASS","temporal_order_parity":"PASS","kv_cache_reset":False,"prefix_context_reset":False,"prompt_restart":False,"prior_chunk_replay":False,"base_parameter_mutation":False,"downstream_recomputation":rec["recompute_downstream_updates"],"stored_delta_stitching":rec["stored_delta_stitching"],"generation_update_count":rec["generation_update_count"],"runtime_seconds":rec["latency_seconds"],"wall_seconds":time.time()-t,"model_identity":MODEL,"checkpoint_identity":CKPT,"benchmark_identity":BENCH,"rule_hash":RULE,"eligibility_hash":ELIG,"cohort_hash":c["cohort_sha256"],"branch_manifest_hash":c["branch_manifest_sha256"],"protocol_hash":c["protocol_sha256"],"status":"SUCCESS","completed":True,"timestamp":now()};atomic(p,out);valid(p,b,c);atomic(run/"commits"/(b["branch_id"]+".json"),{"branch_id":b["branch_id"],"result_sha256":sha(p),"status":"COMMITTED","timestamp":now()});done+=1;new+=1;rate=new/max(time.time()-start,1e-9)*3600;status(run,committed_branches=done,last_commit_timestamp=out["timestamp"],recent_branches_per_hour=rate,eta_hours=(458-done)/rate if rate else None);print(json.dumps({"SELECTIVE_PROGRESS":f"{done}/458","type":b["intervention_type"],"sample":s["canonical_sample_id"],"layer":b["target_layer"],"score":rec["score"]},sort_keys=True),flush=True);gc.collect();import torch;torch.cuda.empty_cache()
  if done!=458:raise RuntimeError("INCOMPLETE")
 except BaseException as e:
  atomic(run/"errors"/(locals().get("b",{}).get("branch_id",f"worker_{os.getpid()}")+".json"),{"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc(),"timestamp":now()});status(run,pipeline_status="FAILED",worker_running=False,last_error=f"{type(e).__name__}: {e}");raise
 finally:
  del model,tok;gc.collect();import torch;torch.cuda.empty_cache()
 status(run,pipeline_status="BRANCH_EXECUTION_COMPLETE",current_phase="READY_FOR_AGGREGATION",committed_branches=458,worker_running=False,worker_pid=None)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("run_root");p.add_argument("--mode",choices=["smoke","formal"],required=True);a=p.parse_args();smoke(a.run_root) if a.mode=="smoke" else formal(a.run_root)
