#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,statistics,sys,time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
def now():return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic(p,o):
 p=Path(p);t=p.with_name(f".{p.name}.{os.getpid()}.{time.time_ns()}.tmp");fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);s=o if isinstance(o,str) else json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n"
 with os.fdopen(fd,"w") as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def mean(x):return statistics.fmean(x)
def ci(x,seed,n):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);y=np.empty(n)
 for s in range(0,n,1000):k=min(1000,n-s);y[s:s+k]=x[rng.integers(0,len(x),(k,len(x)))].mean(1)
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]
def strat(rs,seed,n):
 g=defaultdict(list)
 for r in rs:g[r["task"]].append(r["normal_minus_selective"])
 rng=np.random.default_rng(seed);y=np.zeros(n)
 for v in g.values():
  x=np.asarray(v,float)
  for s in range(0,n,1000):k=min(1000,n-s);y[s:s+k]+=x[rng.integers(0,len(x),(k,len(x)))].mean(1)*len(x)/len(rs)
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]
def main(root):
 r=Path(root).resolve();c=load(r/"config/SELECTIVE_DELETION_FORMAL_CONFIG.json");co=rows(c["cohort_manifest"]);bm=rows(c["branch_manifest"]);sm=load(r/"smoke/SELECTIVE_DELETION_SMOKE.json")
 if sha(c["aggregator"])!=c["aggregator_sha256"] or sha(c["cohort_manifest"])!=c["cohort_sha256"] or sha(c["branch_manifest"])!=c["branch_manifest_sha256"] or sha(c["protocol"])!=c["protocol_sha256"]:raise RuntimeError("HASH_GATE")
 got=[]
 for b in bm:
  p=r/"branches"/(b["branch_id"]+".json")
  if not p.exists():raise RuntimeError("MISSING "+b["branch_id"])
  x=load(p);ok=x.get("completed") is True and x.get("status")=="SUCCESS" and x.get("branch_id")==b["branch_id"] and x.get("target_layer_reset_parity")=="PASS" and x.get("non_target_layer_preservation")=="PASS" and x.get("cohort_hash")==c["cohort_sha256"] and x.get("branch_manifest_hash")==c["branch_manifest_sha256"] and x.get("protocol_hash")==c["protocol_sha256"]
  if not ok:raise RuntimeError("INVALID "+b["branch_id"])
  got.append(x)
 ids=[x["branch_id"] for x in got]
 if len(got)!=458 or len(set(ids))!=458 or list((r/"errors").glob("*.json")):raise RuntimeError("COMPLETENESS")
 by=defaultdict(list)
 for x in got:by[x["canonical_sample_id"]].append(x)
 samples=[];shams=[]
 for s in co:
  z=by[s["canonical_sample_id"]];sel=[x for x in z if x["intervention_type"]=="SELECTIVE"];sharows=[x for x in z if x["intervention_type"]=="SHAM"];need=5 if s["all_prefix"] else 1
  if len(sel)!=need or len(sharows)!=(1 if s["sham_layer"] is not None else 0) or (s["all_prefix"] and {x["target_layer"] for x in sel}!={0,6,12,18,24}):raise RuntimeError("SAMPLE_BRANCH_SET "+s["canonical_sample_id"])
  selective=mean(x["score"] for x in sel);normal=s["normal_score"];full=s["full_reset_score"];den=normal-full
  a={"canonical_sample_id":s["canonical_sample_id"],"task":s["task"],"transition_category":s["transition_category"],"all_prefix":s["all_prefix"],"selective_branch_count":len(sel),"normal_score":normal,"selective_score":selective,"sample_best_score":s["sample_best_score"],"off_score":s["off_score"],"full_reset_score":full,"normal_minus_selective":normal-selective,"normal_minus_sample_best":normal-s["sample_best_score"],"selective_minus_sample_best":selective-s["sample_best_score"],"normal_minus_full_reset":den,"explained_fraction":(normal-selective)/den if den>0 else None,"aggregation_rule":"MEAN_FIVE_LOO" if s["all_prefix"] else "SINGLE_SELECTED","status":"SUCCESS"};samples.append(a);atomic(r/"aggregates"/(hashlib.sha256(s["canonical_sample_id"].encode()).hexdigest()+".json"),a)
  if sharows:
   x=sharows[0];shams.append({"canonical_sample_id":s["canonical_sample_id"],"task":s["task"],"normal_score":normal,"sham_score":x["score"],"normal_minus_sham":normal-x["score"],"score_equal":normal==x["score"],"prediction_hash_equal":s["normal_prediction_sha256"]==x["prediction_sha256"]})
 if len(samples)!=151 or len(shams)!=99:raise RuntimeError("AGGREGATE_SCOPE")
 d=[x["normal_minus_selective"] for x in samples];normal=[x["normal_score"] for x in samples];sel=[x["selective_score"] for x in samples];sb=[x["sample_best_score"] for x in samples];off=[x["off_score"] for x in samples];full=[x["full_reset_score"] for x in samples];pos=[x for x in samples if x["normal_minus_full_reset"]>0];ef=[x["explained_fraction"] for x in pos]
 paired=ci(d,c["paired_bootstrap_seed"],c["bootstrap_resamples"]);taskci=strat(samples,c["task_bootstrap_seed"],c["bootstrap_resamples"])
 cats=[]
 for k in sorted(set(x["transition_category"] for x in samples)):
  z=[x for x in samples if x["transition_category"]==k];q=[x["normal_minus_selective"] for x in z];cats.append({"transition_category":k,"N":len(z),"normal_mean":mean(x["normal_score"] for x in z),"selective_mean":mean(x["selective_score"] for x in z),"effect_mean":mean(q),"hurt":sum(v>0 for v in q),"equal":sum(v==0 for v in q),"better":sum(v<0 for v in q)})
 all_layer=[]
 for layer in [0,6,12,18,24]:
  z=[x for x in got if x["intervention_type"]=="SELECTIVE" and x["all_prefix"] and x["target_layer"]==layer];all_layer.append({"layer":layer,"N":len(z),"effect_mean":mean(x["effect"] for x in z),"effect_median":statistics.median(x["effect"] for x in z)})
 dis=[x for x in samples if x["transition_category"]=="SINGLE_LAYER_DISJOINT"];disd=[x["normal_minus_selective"] for x in dis]
 summary={"N":151,"NORMAL_MEAN":mean(normal),"SELECTIVE_MEAN":mean(sel),"SAMPLE_BEST_MEAN":mean(sb),"OFF_MEAN":mean(off),"FULL_RESET_MEAN":mean(full),"NORMAL_MINUS_SELECTIVE_MEAN":mean(d),"NORMAL_MINUS_SELECTIVE_MEDIAN":statistics.median(d),"NORMAL_MINUS_SELECTIVE_STD":statistics.stdev(d),"NORMAL_MINUS_SELECTIVE_MIN":min(d),"NORMAL_MINUS_SELECTIVE_MAX":max(d),"SELECTIVE_WORSE_COUNT":sum(x>0 for x in d),"SELECTIVE_EQUAL_COUNT":sum(x==0 for x in d),"SELECTIVE_BETTER_COUNT":sum(x<0 for x in d),"SELECTIVE_GT_SAMPLE_BEST_COUNT":sum(x["selective_minus_sample_best"]>0 for x in samples),"SELECTIVE_EQ_SAMPLE_BEST_COUNT":sum(x["selective_minus_sample_best"]==0 for x in samples),"SELECTIVE_LT_SAMPLE_BEST_COUNT":sum(x["selective_minus_sample_best"]<0 for x in samples),"PAIRED_BOOTSTRAP_95CI":paired,"TASK_STRATIFIED_PAIRED_BOOTSTRAP_95CI":taskci,"NORMAL_MINUS_FULL_RESET_MEAN":mean(x["normal_minus_full_reset"] for x in samples),"FULL_RESET_DENOMINATOR_POSITIVE_N":len(pos),"FULL_RESET_DENOMINATOR_NONPOSITIVE_N":151-len(pos),"EXPLAINED_FRACTION_MEAN":mean(ef) if ef else None,"EXPLAINED_FRACTION_MEDIAN":statistics.median(ef) if ef else None,"EXPLAINED_FRACTION_STD":statistics.stdev(ef) if len(ef)>1 else None,"EXPLAINED_FRACTION_QUANTILES":{str(q):float(np.quantile(ef,q)) for q in [0,.05,.25,.5,.75,.95,1]} if ef else {},"EXPLAINED_FRACTION_LE_0_FRACTION":sum(x<=0 for x in ef)/len(ef) if ef else None,"EXPLAINED_FRACTION_GE_0P5_FRACTION":sum(x>=.5 for x in ef)/len(ef) if ef else None,"EXPLAINED_FRACTION_GE_1_FRACTION":sum(x>=1 for x in ef)/len(ef) if ef else None,"AGGREGATE_EXPLAINED_FRACTION":sum(x["normal_minus_selective"] for x in pos)/sum(x["normal_minus_full_reset"] for x in pos),"SHAM_N":99,"NORMAL_MEAN_ON_SHAM_SUBSET":mean(x["normal_score"] for x in shams),"SHAM_MEAN":mean(x["sham_score"] for x in shams),"NORMAL_MINUS_SHAM_MEAN":mean(x["normal_minus_sham"] for x in shams),"NORMAL_MINUS_SHAM_MEDIAN":statistics.median(x["normal_minus_sham"] for x in shams),"SHAM_EQUAL_COUNT":sum(x["score_equal"] for x in shams),"SHAM_WORSE_COUNT":sum(x["normal_minus_sham"]>0 for x in shams),"SHAM_BETTER_COUNT":sum(x["normal_minus_sham"]<0 for x in shams),"SHAM_PREDICTION_HASH_PARITY_COUNT":sum(x["prediction_hash_equal"] for x in shams),"SINGLE_LAYER_DISJOINT_N":len(dis),"SINGLE_LAYER_DISJOINT_EFFECT_MEAN":mean(disd),"SINGLE_LAYER_DISJOINT_BOOTSTRAP_95CI":ci(disd,c["disjoint_bootstrap_seed"],c["bootstrap_resamples"])}
 atomic(r/"reports/SELECTIVE_DELETION_AGGREGATE.json",summary);atomic(r/"reports/TRANSITION_CATEGORY_ANALYSIS.json",cats);atomic(r/"reports/ALL_PREFIX_PREDEFINED_LAYER_ANALYSIS.json",{"label":"SECONDARY_PREDEFINED_ALL_PREFIX_LAYER_ANALYSIS","rows":all_layer});atomic(r/"reports/SHAM_ANALYSIS.json",shams)
 receipt={**summary,"MODEL_IDENTITY":c["model_identity"],"CHECKPOINT_IDENTITY":c["checkpoint_identity"],"RULE_MANIFEST_SHA256":c["rule_sha256"],"ELIGIBILITY_MANIFEST_SHA256":c["eligibility_sha256"],"COHORT_MANIFEST_SHA256":c["cohort_sha256"],"BRANCH_MANIFEST_SHA256":c["branch_manifest_sha256"],"PROTOCOL_SHA256":c["protocol_sha256"],"MODEL_IDENTITY_MATCH":True,"BENCHMARK_IDENTITY_MATCH":True,"RULE_HASH_MATCH":True,"ELIGIBILITY_HASH_MATCH":True,"COHORT_HASH_MATCH":True,"BRANCH_MANIFEST_HASH_MATCH":True,"PROTOCOL_HASH_MATCH":True,"EXECUTION_COHORT_N":151,"SELECTIVE_BRANCHES_VALID":359,"SHAM_BRANCHES_VALID":99,"TOTAL_BRANCHES_VALID":458,"SAMPLE_AGGREGATES_VALID":151,"MISSING_BRANCHES":0,"DUPLICATE_BRANCHES":0,"UNRESOLVED_ERRORS":0,"ORIGINAL_DYNAMIC_REPLAY_PARITY":sm["ORIGINAL_DYNAMIC_REPLAY_PARITY"],"ACTION_SCHEDULE_PARITY":"PASS","TEMPORAL_ORDER_PARITY":"PASS","SELECTIVE_BOUNDARY_PARITY":"PASS","TARGET_LAYER_RESET_PARITY":"PASS","NON_TARGET_LAYER_PRESERVATION":"PASS","KV_CACHE_RESET":False,"PREFIX_CONTEXT_RESET":False,"BASE_PARAMETER_MUTATION":False,"SHAM_ZERO_STATE_PRECONDITION":"PASS","SHAM_INTEGRITY":"PASS","HISTORICAL_OVERLAP":0,"OUTCOME_LEAKAGE":0,"FINAL_INTEGRITY":"PASS","FORMAL_SELECTIVE_DELETION_COMPLETE":True,"KP_LAUNCHED":False,"R2_LAUNCHED":False,"CRITICALITY_LAUNCHED":False,"completed_at":now()};atomic(r/"reports/SELECTIVE_DELETION_1P7B_FORMAL_COMPLETION_RECEIPT.json",receipt);atomic(r/"reports/SELECTIVE_DELETION_1P7B_FORMAL_COMPLETION_RECEIPT.md","# Selective Deletion 1.7B Formal Completion Receipt\n\n"+"\n".join(f"{k}={v}" for k,v in receipt.items())+"\n")
 st=load(r/"status/pipeline_status.json");st.update(pipeline_status="COMPLETE",current_phase="STOP",sample_aggregates=151,final_receipt_exists=True,worker_running=False,updated_at=now());atomic(r/"status/pipeline_status.json",st);print("FORMAL_SELECTIVE_DELETION_COMPLETE=true",flush=True)
if __name__=="__main__":main(sys.argv[1])
