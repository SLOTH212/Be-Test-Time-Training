#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, math, os, time, traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

MODEL_SHA="ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f"
CKPT_SHA="f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd"
BENCH_SHA="5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5"
RULE_SHA="e816801b71358265c6c9458e8754887e71508c6b26448e6dad5a336f64fecf47"
ELIG_SHA="e74831509b3d46068de7364aeffeead89003d057d086e23a496ab3c3f2e55220"
EXEC_SHA="293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e"
SCORER_SHA="cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556"
BACKEND_SHA="c15b6872c625ee13ee2b71e522a9414e6f797edd6a695e185465dff05c5326ee"
LAYERS=[0,6,12,18,24]; ACTIONS=["OFF","L0","L6","L12","L18","L24","ALL"]
def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic(p,obj):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f".{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
 fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  with os.fdopen(fd,"w") as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
  os.replace(t,p); d=os.open(p.parent,os.O_RDONLY); os.fsync(d); os.close(d)
 except BaseException:
  try: os.unlink(t)
  except FileNotFoundError: pass
  raise
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def name(s): return hashlib.sha256(s.encode()).hexdigest()+".json"
def status(run,**kw):
 p=run/"status/pipeline_status.json"; x=load(p) if p.exists() else {}; x.update(kw); x["updated_at"]=now(); atomic(p,x)

class Ctx:
 def __init__(self,cfg):
  self.run=Path(cfg["run_root"]); self.code=Path("/home/USER/ttt/code/ntp"); self.ckpt=Path(cfg["model_root"])
  self.score_fn=imp(Path(cfg["scorer"]),"reset_scorer").per_sample_score
  self.rows=[]
  for i,r in enumerate(rows(Path(cfg["benchmark_root"])/"sample_manifest.jsonl")): r["_sample_index"]=i; self.rows.append(r)
  self.by_id={r["sample_id"]:r for r in self.rows}; self.manifest={"tasks":sorted({r["task"] for r in self.rows})}
  self.baseline={a:{} for a in ACTIONS}
  for r in rows(Path(cfg["fixed7_root"])/"manifests/all_fixed7_results.jsonl"): r["official_sample_score"]=float(r["score"]); self.baseline[r["mode"]][r["sample_id"]]=r
  if len(self.by_id)!=6500 or any(len(self.baseline[a])!=6500 for a in ACTIONS): raise RuntimeError("CONTEXT_AUTHORITY_INVALID")

def gates(run):
 run=Path(run).resolve(); cfg=load(run/"config/STATE_RESET_FORMAL_CONFIG.json")
 if sorted(os.sched_getaffinity(0))!=list(range(8,24)): raise RuntimeError("CPU_AFFINITY_MISMATCH")
 checks={"model":sha(Path(cfg["model_root"])/"model.safetensors")==MODEL_SHA,"benchmark":sha(Path(cfg["benchmark_root"])/"benchmark_manifest.json")==BENCH_SHA,
 "rule":sha(cfg["rule_manifest"])==RULE_SHA,"eligibility":sha(cfg["eligibility_manifest"])==ELIG_SHA,"cohort":sha(cfg["cohort_manifest"])==cfg["cohort_sha256"],
 "protocol":sha(cfg["protocol"])==cfg["protocol_sha256"],"amendment":sha(cfg["amendment"])==cfg["amendment_sha256"],"executor":sha(cfg["executor"])==EXEC_SHA,
 "scorer":sha(cfg["scorer"])==SCORER_SHA,"backend":sha(cfg["backend"])==BACKEND_SHA,"worker":sha(cfg["worker"])==cfg["worker_sha256"],"aggregator":sha(cfg["aggregator"])==cfg["aggregator_sha256"],"controller":sha(cfg["controller"])==cfg["controller_sha256"]}
 if not all(checks.values()): raise RuntimeError("AUTHORITY_HASH_MISMATCH "+json.dumps(checks,sort_keys=True))
 cohort=rows(cfg["cohort_manifest"])
 if len(cohort)!=167 or len({r["canonical_sample_id"] for r in cohort})!=167: raise RuntimeError("COHORT_SCOPE_MISMATCH")
 for r in cohort:
  a=r["original_dynamic_actions"]; b=r["reset_boundary_chunk_index"]
  if len(a)!=r["full_chunk_count"] or b!=max(i+1 for i in range(1,len(a)) if a[i]!=a[i-1]): raise RuntimeError("BOUNDARY_INVARIANT")
  if r["dynamic_minus_sample_best"]<=0 or r["historical_overlap"]: raise RuntimeError("COHORT_INVARIANT")
 return run,cfg,cohort,checks
def configure(model,b):
 for i in LAYERS:
  m=model.model.layers[i].mlp; m._reset_boundary_chunk=b; m._instrumentation_enabled=True; m._state_events=[]; m._reset_events=[]
def eval_one(runner,backend,ctx,model,tok,row,stage,boundary,ordinal,total,capture=False):
 configure(model,boundary); ev=runner.Evaluator(ctx,model,tok,ctx.by_id[row["canonical_sample_id"]],stage,ordinal,total,ctx.run/"status/worker_internal_progress.json")
 rec=ev.evaluate(tuple(row["original_dynamic_actions"]),capture=capture,force=True); ev.finish()
 se=[]; re=[]
 for i in LAYERS:
  m=model.model.layers[i].mlp; se += backend.materialize_scalar(list(m._state_events)); re += backend.materialize_scalar(list(m._reset_events))
 if len(se)!=5*row["full_chunk_count"] or len(re)!=(5 if boundary is not None else 0): raise RuntimeError("INSTRUMENTATION_EVENT_COUNT")
 if any(e["selected_action"]!=row["original_dynamic_actions"][e["chunk_index"]-1] for e in se): raise RuntimeError("ACTION_SCHEDULE_MISMATCH")
 return rec,se,re
def runtime(run,cfg):
 runner=imp(cfg["executor"],"frozen_state_reset_executor"); backend=imp(cfg["backend"],"historical_state_reset_backend"); ctx=Ctx(cfg)
 import torch; torch.set_num_threads(16); status(run,current_phase="MODEL_LOADING",worker_pid=os.getpid()); model,tok=runner.load_model(ctx); base=runner.base_hash(model)
 backend.install_instrumented_dynamic_forward(model,runner.tensor_hash)
 atomic(run/"audit/MODEL_LOAD_AUDIT.json",{"status":"PASS","model_identity":MODEL_SHA,"checkpoint_identity":CKPT_SHA,"base_weight_runtime_hash":base,"cuda_device":torch.cuda.get_device_name(0),"timestamp":now()})
 return runner,backend,ctx,model,tok,base
def valid(p,row,cfg):
 r=load(p)
 ok=r.get("completed") is True and r.get("status")=="SUCCESS" and r.get("canonical_sample_id")==row["canonical_sample_id"] and r.get("cohort_hash")==cfg["cohort_sha256"] and r.get("protocol_hash")==cfg["protocol_sha256"] and r.get("reset_applied") is True and r.get("reset_count")==1 and r.get("reset_to_checkpoint_parity")=="PASS" and r.get("base_parameter_mutation") is False
 if not ok: raise RuntimeError("INVALID_EXISTING_COMMIT "+str(p)); return r
 return r
def smoke(run):
 run,cfg,cohort,checks=gates(run)
 if list((run/"results").glob("*.json")): raise RuntimeError("FORMAL_RESULTS_EXIST_BEFORE_SMOKE")
 selected=[cohort[i-1] for i in cfg["smoke_orders"]]; status(run,pipeline_status="SMOKE_RUNNING",current_phase="NATIVE_RESET_SMOKE",last_error="NONE")
 runner,backend,ctx,model,tok,base=runtime(run,cfg); out=[]
 try:
  for j,row in enumerate(selected,1):
   native,_,nr=eval_one(runner,backend,ctx,model,tok,row,"state_reset_native_smoke",None,j,3,True)
   reset,se,re=eval_one(runner,backend,ctx,model,tok,row,"state_reset_reset_smoke",row["reset_boundary_chunk_index"],j,3,False)
   again,_,_=eval_one(runner,backend,ctx,model,tok,row,"state_reset_native_repeat_smoke",None,j,3,True)
   parity=native["score"]==row["native_dynamic_score"] and native["prediction_hash"]==row["native_prediction_sha256"]
   repeat=native["score"]==again["score"] and native["prediction_hash"]==again["prediction_hash"]
   residual=max(e["post_reset_max_abs_residual"] for e in re)
   boundary=all(e["chunk_index"]==row["reset_boundary_chunk_index"] for e in re)
   preserved=reset["sequence_hash"]==native["sequence_hash"] and reset["prompt_token_count"]==native["prompt_token_count"] and reset["generation_update_count"]==native["generation_update_count"]==0
   base_ok=runner.base_hash(model)==base
   if not (parity and repeat and residual==0 and boundary and preserved and base_ok): raise RuntimeError("SMOKE_GATE_FAILED "+row["canonical_sample_id"])
   out.append({"canonical_sample_id":row["canonical_sample_id"],"native_score":native["score"],"reset_score":reset["score"],"native_prediction_parity":parity,"sample_reset_parity":repeat,"reset_boundary_observed":row["reset_boundary_chunk_index"],"per_layer_reset_events":re,"reset_max_abs_residual":residual,"action_schedule_parity":True,"preserved_context_semantics":True,"base_parameter_mutation":False})
 finally:
  del model,tok; gc.collect(); import torch; torch.cuda.empty_cache()
 s={"status":"PASS","formal_result":False,"smoke_n":3,"ORIGINAL_DYNAMIC_REPLAY_PARITY":"PASS","ACTION_SCHEDULE_PARITY":"PASS","RESET_BOUNDARY_PARITY":"PASS","RESET_TO_CHECKPOINT_PARITY":"PASS","PRESERVED_CONTEXT_SEMANTICS":"PASS","BASE_PARAMETER_MUTATION":False,"SAMPLE_RESET_PARITY":"PASS","rows":out,"authority_checks":checks,"completed_at":now()}
 atomic(run/"smoke/STATE_RESET_SMOKE.json",s); status(run,pipeline_status="SMOKE_PASS",current_phase="READY_FOR_FORMAL_LAUNCH",worker_pid=None,smoke_status="PASS"); print(json.dumps(s,sort_keys=True),flush=True)
def formal(run):
 run,cfg,cohort,_=gates(run); sm=load(run/"smoke/STATE_RESET_SMOKE.json")
 if sm["status"]!="PASS": raise RuntimeError("SMOKE_GATE_NOT_PASS")
 done=sum((run/"results"/name(r["canonical_sample_id"])).exists() for r in cohort); status(run,pipeline_status="RUNNING",current_phase="FORMAL_STATE_RESET",committed_results=done,total_expected=167,worker_pid=os.getpid(),worker_running=True,formal_launched=True,last_error="NONE")
 runner,backend,ctx,model,tok,base=runtime(run,cfg); started=time.time(); new=0
 try:
  for row in cohort:
   sid=row["canonical_sample_id"]; p=run/"results"/name(sid)
   if p.exists(): valid(p,row,cfg); continue
   claim={"canonical_sample_id":sid,"state":"running","worker_pid":os.getpid(),"timestamp":now()}; atomic(run/"claims"/name(sid),claim); status(run,current_sample=sid,current_task=row["task"],current_reset_boundary=row["reset_boundary_chunk_index"])
   t=time.time(); rec,se,re=eval_one(runner,backend,ctx,model,tok,row,"formal_state_reset_1p7b_v1",row["reset_boundary_chunk_index"],row["cohort_order"],167)
   if runner.base_hash(model)!=base: raise RuntimeError("BASE_WEIGHT_MUTATION "+sid)
   residual=max(e["post_reset_max_abs_residual"] for e in re); boundary=all(e["chunk_index"]==row["reset_boundary_chunk_index"] for e in re)
   if residual!=0 or not boundary: raise RuntimeError("RESET_INTEGRITY_FAILURE "+sid)
   native=float(row["native_dynamic_score"]); reset=float(rec["score"]); sb=float(row["sample_best_score"]); gain=native-sb
   out={**{k:row[k] for k in ["canonical_sample_id","task","benchmark_global_index","cohort_order","original_dynamic_actions","full_chunk_count","reset_boundary_chunk_index","terminal_action","switch_count"]},"native_dynamic_score":native,"reset_score":reset,"sample_best_score":sb,"off_score":float(row["off_score"]),"native_minus_reset":native-reset,"native_minus_sample_best":gain,"reset_minus_sample_best":reset-sb,"retained_gain":(reset-sb)/gain,"reset_applied":True,"reset_count":1,"reset_boundary_observed":row["reset_boundary_chunk_index"],"candidate_layers_reset":LAYERS,"per_layer_reset_event_count":len(re),"reset_max_abs_residual":residual,"reset_to_checkpoint_parity":"PASS","action_schedule_parity":"PASS","base_parameter_mutation":False,"prompt_restart":False,"prior_chunk_replay":False,"kv_cache_reset":False,"prefix_context_reset":False,"rng_reset":False,"generation_change":False,"recompute_downstream_updates":rec["recompute_downstream_updates"],"stored_delta_stitching":rec["stored_delta_stitching"],"generation_update_count":rec["generation_update_count"],"prediction_sha256":rec["prediction_hash"],"runtime_seconds":rec["latency_seconds"],"wall_seconds":time.time()-t,"model_identity":MODEL_SHA,"checkpoint_identity":CKPT_SHA,"benchmark_identity":BENCH_SHA,"rule_hash":RULE_SHA,"eligibility_hash":ELIG_SHA,"cohort_hash":cfg["cohort_sha256"],"protocol_hash":cfg["protocol_sha256"],"status":"SUCCESS","completed":True,"timestamp":now()}
   atomic(p,out); valid(p,row,cfg); atomic(run/"commits"/name(sid),{"canonical_sample_id":sid,"result_sha256":sha(p),"status":"COMMITTED","timestamp":now()}); claim.update(state="committed",result_sha256=sha(p)); atomic(run/"claims"/name(sid),claim)
   done+=1; new+=1; rate=new/max(time.time()-started,1e-9)*3600; status(run,committed_results=done,last_commit_timestamp=out["timestamp"],recent_samples_per_hour=rate,eta_hours=(167-done)/rate if rate else None)
   print(json.dumps({"STATE_RESET_PROGRESS":f"{done}/167","sample_id":sid,"native":native,"reset":reset,"runtime_seconds":rec["latency_seconds"]},sort_keys=True),flush=True); gc.collect(); import torch; torch.cuda.empty_cache()
  if done!=167: raise RuntimeError("INCOMPLETE")
 except BaseException as e:
  atomic(run/"errors"/(name(locals().get("sid","worker"))),{"failure_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"timestamp":now()}); status(run,pipeline_status="FAILED",current_phase="FORMAL_FAILED",worker_running=False,last_error=f"{type(e).__name__}: {e}"); raise
 finally:
  del model,tok; gc.collect(); import torch; torch.cuda.empty_cache()
 status(run,pipeline_status="FORMAL_EXECUTION_COMPLETE",current_phase="READY_FOR_AGGREGATION",committed_results=167,worker_pid=None,worker_running=False)
if __name__=="__main__":
 a=argparse.ArgumentParser(); a.add_argument("run_root"); a.add_argument("--mode",choices=["smoke","formal"],required=True); z=a.parse_args(); smoke(z.run_root) if z.mode=="smoke" else formal(z.run_root)
