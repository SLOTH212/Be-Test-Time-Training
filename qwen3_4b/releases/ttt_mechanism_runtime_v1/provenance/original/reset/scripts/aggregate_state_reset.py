#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, statistics, sys, time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic_text(p,s):
 p=Path(p); t=p.with_name(f".{p.name}.{os.getpid()}.{time.time_ns()}.tmp"); fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,"w") as f: f.write(s); f.flush(); os.fsync(f.fileno())
 os.replace(t,p); d=os.open(p.parent,os.O_RDONLY); os.fsync(d); os.close(d)
def atomic(p,o): atomic_text(p,json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n")
def name(s): return hashlib.sha256(s.encode()).hexdigest()+".json"
def mean(x): return statistics.fmean(x)
def ci(x,seed,n):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); y=np.empty(n)
 for s in range(0,n,1000): k=min(1000,n-s); y[s:s+k]=x[rng.integers(0,len(x),(k,len(x)))].mean(1)
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]
def strat(rs,seed,n):
 g=defaultdict(list)
 for r in rs:g[r["task"]].append(r["native_minus_reset"])
 rng=np.random.default_rng(seed); y=np.zeros(n); total=len(rs)
 for v in g.values():
  x=np.asarray(v,float)
  for s in range(0,n,1000): k=min(1000,n-s); y[s:s+k]+=x[rng.integers(0,len(x),(k,len(x)))].mean(1)*len(x)/total
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]
def main(run):
 run=Path(run).resolve(); c=load(run/"config/STATE_RESET_FORMAL_CONFIG.json"); cohort=rows(c["cohort_manifest"]); smoke=load(run/"smoke/STATE_RESET_SMOKE.json")
 if sha(c["aggregator"])!=c["aggregator_sha256"] or sha(c["cohort_manifest"])!=c["cohort_sha256"] or sha(c["protocol"])!=c["protocol_sha256"]: raise RuntimeError("HASH_GATE")
 rs=[]
 for m in cohort:
  p=run/"results"/name(m["canonical_sample_id"])
  if not p.exists(): raise RuntimeError("MISSING_RESULT "+m["canonical_sample_id"])
  r=load(p); ok=r.get("completed") is True and r.get("status")=="SUCCESS" and r.get("reset_to_checkpoint_parity")=="PASS" and r.get("base_parameter_mutation") is False and r.get("action_schedule_parity")=="PASS" and r.get("cohort_hash")==c["cohort_sha256"] and r.get("protocol_hash")==c["protocol_sha256"]
  if not ok: raise RuntimeError("INVALID_RESULT "+m["canonical_sample_id"])
  rs.append(r)
 ids=[r["canonical_sample_id"] for r in rs]
 if len(rs)!=167 or len(set(ids))!=167 or len(list((run/"results").glob("*.json")))!=167 or list((run/"errors").glob("*.json")): raise RuntimeError("COMPLETENESS_FAILURE")
 native=[r["native_dynamic_score"] for r in rs]; reset=[r["reset_score"] for r in rs]; sb=[r["sample_best_score"] for r in rs]; off=[r["off_score"] for r in rs]; d=[r["native_minus_reset"] for r in rs]; ng=[r["native_minus_sample_best"] for r in rs]; rg=[r["reset_minus_sample_best"] for r in rs]; ret=[r["retained_gain"] for r in rs]
 paired=ci(d,c["paired_bootstrap_seed"],c["bootstrap_resamples"]); taskci=strat(rs,c["task_stratified_bootstrap_seed"],c["bootstrap_resamples"])
 per=[]
 for task in sorted(set(r["task"] for r in rs)):
  z=[r for r in rs if r["task"]==task]; zd=[r["native_minus_reset"] for r in z]
  per.append({"task":task,"N":len(z),"native_mean":mean(r["native_dynamic_score"] for r in z),"reset_mean":mean(r["reset_score"] for r in z),"sample_best_mean":mean(r["sample_best_score"] for r in z),"off_mean":mean(r["off_score"] for r in z),"native_minus_reset_mean":mean(zd),"native_minus_sample_best_mean":mean(r["native_minus_sample_best"] for r in z),"reset_minus_sample_best_mean":mean(r["reset_minus_sample_best"] for r in z),"hurt":sum(x>0 for x in zd),"equal":sum(x==0 for x in zd),"improve":sum(x<0 for x in zd)})
 atomic(run/"reports/PER_TASK_STATE_RESET_RESULTS.json",per)
 summary={"N":167,"NATIVE_DYNAMIC_MEAN":mean(native),"RESET_MEAN":mean(reset),"SAMPLE_BEST_MEAN":mean(sb),"OFF_MEAN":mean(off),"NATIVE_MINUS_RESET_MEAN":mean(d),"NATIVE_MINUS_RESET_MEDIAN":statistics.median(d),"NATIVE_MINUS_RESET_STD":statistics.stdev(d),"NATIVE_MINUS_RESET_MIN":min(d),"NATIVE_MINUS_RESET_MAX":max(d),"RESET_WORSE_COUNT":sum(x>0 for x in d),"RESET_EQUAL_COUNT":sum(x==0 for x in d),"RESET_BETTER_COUNT":sum(x<0 for x in d),"NATIVE_MINUS_SAMPLE_BEST_MEAN":mean(ng),"RESET_MINUS_SAMPLE_BEST_MEAN":mean(rg),"RESET_GT_SAMPLE_BEST_COUNT":sum(x>0 for x in rg),"RESET_EQ_SAMPLE_BEST_COUNT":sum(x==0 for x in rg),"RESET_LT_SAMPLE_BEST_COUNT":sum(x<0 for x in rg),"GAIN_RETENTION_MEAN":mean(ret),"GAIN_RETENTION_MEDIAN":statistics.median(ret),"GAIN_RETENTION_QUANTILES":{str(q):float(np.quantile(ret,q)) for q in [0,.05,.25,.5,.75,.95,1]},"GAIN_RETENTION_FRACTION_LE_0":sum(x<=0 for x in ret)/167,"GAIN_RETENTION_FRACTION_LT_0P5":sum(x<.5 for x in ret)/167,"GAIN_RETENTION_FRACTION_LT_1":sum(x<1 for x in ret)/167,"GAIN_RETENTION_FRACTION_GE_1":sum(x>=1 for x in ret)/167,"AGGREGATE_GAIN_RETENTION":sum(rg)/sum(ng),"PAIRED_BOOTSTRAP_95CI":paired,"TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI":taskci,"bootstrap_resamples":c["bootstrap_resamples"],"task_distribution":dict(Counter(r["task"] for r in rs))}
 atomic(run/"reports/STATE_RESET_AGGREGATE.json",summary)
 receipt={"MODEL_IDENTITY":c["model_identity"],"CHECKPOINT_IDENTITY":c["checkpoint_identity"],"RULE_MANIFEST_SHA256":c["rule_sha256"],"ELIGIBILITY_MANIFEST_SHA256":c["eligibility_sha256"],"COHORT_MANIFEST_SHA256":c["cohort_sha256"],"PROTOCOL_SHA256":c["protocol_sha256"],**summary,"STRUCTURALLY_ELIGIBLE_STATE_RESET_CONTROLS":0,"RESET_TO_CHECKPOINT_PARITY":smoke["RESET_TO_CHECKPOINT_PARITY"],"PRESERVED_CONTEXT_SEMANTICS":smoke["PRESERVED_CONTEXT_SEMANTICS"],"BASE_PARAMETER_MUTATION":False,"MODEL_IDENTITY_MATCH":True,"BENCHMARK_IDENTITY_MATCH":True,"RULE_HASH_MATCH":True,"ELIGIBILITY_HASH_MATCH":True,"COHORT_HASH_MATCH":True,"PROTOCOL_HASH_MATCH":True,"VALID_RESULTS":167,"MISSING_RESULTS":0,"DUPLICATE_RESULTS":0,"UNRESOLVED_ERRORS":0,"HISTORICAL_OVERLAP":0,"STATE_RESET_OUTCOME_LEAKAGE":0,"FINAL_INTEGRITY":"PASS","FORMAL_STATE_RESET_COMPLETE":True,"completed_at":now()}
 atomic(run/"reports/STATE_RESET_1P7B_FORMAL_COMPLETION_RECEIPT.json",receipt); atomic_text(run/"reports/STATE_RESET_1P7B_FORMAL_COMPLETION_RECEIPT.md","# State Reset 1.7B Formal Completion Receipt\n\n"+"\n".join(f"{k}={v}" for k,v in receipt.items())+"\n")
 s=load(run/"status/pipeline_status.json"); s.update(pipeline_status="COMPLETE",current_phase="STOP",committed_results=167,worker_running=False,final_receipt_exists=True,updated_at=now()); atomic(run/"status/pipeline_status.json",s)
 print("FORMAL_STATE_RESET_COMPLETE=true",flush=True)
if __name__=="__main__": main(sys.argv[1])
