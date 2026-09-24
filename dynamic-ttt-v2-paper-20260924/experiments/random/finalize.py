import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
R=Path(__file__).resolve().parent
P=json.loads((R/'RANDOM_PROTOCOL.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def put(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def jl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def tsv(n,rows):
 with (R/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
assert not list(R.glob('FAILURE*.json'))
assert sha(R/'RANDOM_PLAN.jsonl')==P['RANDOM_PLAN_SHA256']
for n,h in json.loads((R/'FROZEN_FILES.json').read_text()).items():assert sha(R/n)==h,n
for n,h in json.loads((R/'AUTHORITY_HASHES.json').read_text()).items():assert sha(n)==h,n
assert sha(Path(P['model_path'])/'model.safetensors')==P['CHECKPOINT_IDENTITY']
plan=jl(R/'RANDOM_PLAN.jsonl');expected={(x['sample_id'],x['trajectory_index']):x for x in plan};assert len(expected)==len(plan)==P['total_budget']
raw=[];starts=collections.Counter();ends=collections.Counter()
for w in range(2):
 assert json.loads((R/f'COMPLETE_{w}.json').read_text())['status']=='COMPLETE'
 raw+=jl(R/f'WORKER_RESULTS_{w}.jsonl')
 for e in jl(R/f'EXECUTION_LEDGER_{w}.jsonl'):
  k=(e['sample_id'],e['trajectory_index']);assert e['trajectory_hash']==expected[k]['trajectory_hash']
  (starts if e['event']=='START' else ends)[k]+=1
assert len(raw)==len(plan) and starts==ends==collections.Counter({k:1 for k in expected})
observed={}
for x in raw:
 k=x['sample_id'],x['trajectory_index'];assert k not in observed and k in expected;assert x['trajectory']==expected[k]['trajectory'] and x['trajectory_hash']==expected[k]['trajectory_hash'];assert x['checkpoint_sha256']==P['CHECKPOINT_IDENTITY'] and 0<=x['score']<=1;observed[k]=x
raw.sort(key=lambda x:(x['sample_id'],x['trajectory_index']));(R/'RANDOM_RESULTS.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in raw))
co=list(csv.DictReader((R/'RANDOM_BASELINE_COHORT.tsv').open(),delimiter='\t'));samples={x['sample_id']:x for x in (json.loads(p.read_text()) for p in (R/'samples').glob('*.json'))};assert len(samples)==len(co)==170 and set(samples)=={x['sample_id'] for x in co};group=collections.defaultdict(list)
for x in raw:group[x['sample_id']].append(x)
rows=[]
for c in sorted(co,key=lambda x:x['sample_id']):
 sid=c['sample_id'];x=samples[sid];rs=group[sid];assert len(rs)==x['B_i'];sb=float(c['sample_best_score']);d=float(c['dynamic_score']);rand=max(sb,max(z['score'] for z in rs));assert rand==x['random_score'] and rand>=sb and d>sb
 rows.append(dict(sample_id=sid,task=c['task'],B_i=len(rs),SB=sb,Dynamic=d,Random=rand,dynamic_gain=d-sb,random_gain=rand-sb,dynamic_minus_random=d-rand,random_winner_source=x['winner_source'],random_winner_trajectory=json.dumps(x['winner_trajectory']),random_winner_switches=x['winner_structure']['switch_count'],dynamic_winner_switches=x['dynamic_structure']['switch_count']))
tsv('RANDOM_SAMPLE_SUMMARY.tsv',rows)
def stats(rs):
 n=len(rs);sb=sum(x['SB'] for x in rs)/n;d=sum(x['Dynamic'] for x in rs)/n;rr=sum(x['Random'] for x in rs)/n;dg=sum(x['dynamic_gain'] for x in rs)/n;rg=sum(x['random_gain'] for x in rs)/n;assert dg>0
 return dict(N=n,SB_MEAN=sb*100,DYNAMIC_MEAN=d*100,RANDOM_MEAN=rr*100,DYNAMIC_GAIN_PP=dg*100,RANDOM_GAIN_PP=rg*100,DYNAMIC_MINUS_RANDOM_PP=sum(x['dynamic_minus_random'] for x in rs)/n*100,RANDOM_GAIN_RECOVERY_RATIO=rg/dg,RANDOM_GT_SB_N=sum(x['Random']>x['SB'] for x in rs),RANDOM_EQ_SB_N=sum(x['Random']==x['SB'] for x in rs),RANDOM_GT_DYNAMIC_N=sum(x['Random']>x['Dynamic'] for x in rs),RANDOM_EQ_DYNAMIC_N=sum(x['Random']==x['Dynamic'] for x in rs),RANDOM_LT_DYNAMIC_N=sum(x['Random']<x['Dynamic'] for x in rs))
s=stats(rows);ts=[dict(task=t,**stats([x for x in rows if x['task']==t])) for t in sorted({x['task'] for x in rows})];tsv('RANDOM_TASK_SUMMARY.tsv',ts)
inputs=[dict(sample_id=x['sample_id'],task=x['task'],difference=x['dynamic_minus_random']) for x in sorted(rows,key=lambda x:(x['task'],x['sample_id']))];put('BOOTSTRAP_INPUT.json',inputs);rng=np.random.default_rng(20260923);sums=np.zeros(20000);drawhash=hashlib.sha256()
for task in sorted({x['task'] for x in inputs}):
 vals=np.array([x['difference'] for x in inputs if x['task']==task]);n=len(vals);ix=rng.integers(0,n,size=(20000,n),dtype=np.int64);drawhash.update(task.encode()+b'\0');drawhash.update(ix.astype('<i8').tobytes());sums+=vals[ix].sum(axis=1)
ci=np.quantile(sums/len(rows)*100,[.025,.975],method='linear').tolist();s['DYNAMIC_MINUS_RANDOM_CI']=ci
put('RANDOM_BOOTSTRAP_AUDIT.json',dict(seed=20260923,draws=20000,numpy=np.__version__,RNG='PCG64',input_sha256=sha(R/'BOOTSTRAP_INPUT.json'),ordering='task lexicographic then sample_id lexicographic',draw_hash=drawhash.hexdigest(),draw_hash_encoding='task UTF8 + null then little-endian int64 index matrix per task',CI95=ci,quantile='linear',sample_weighted=True,random_seed_uncertainty_estimated=False,interpretation='Sample uncertainty conditional on this fixed random realization; not expected random-search performance CI'))
def structural(xs):
 return dict(n=len(xs),switch_count_histogram=dict(sorted(collections.Counter(x['switch_count'] for x in xs).items())),distinct_action_count_histogram=dict(sorted(collections.Counter(x['distinct_action_count'] for x in xs).items())),first_action_counts=dict(collections.Counter(x['first_action'] for x in xs)),last_action_counts=dict(collections.Counter(x['last_action'] for x in xs)))
put('RANDOM_STRUCTURE.json',dict(all_evaluated=structural(raw),random_only_best=structural([x['best_nonconstant_structure'] for x in samples.values()]),selected_winner_including_shared_constants=structural([x['winner_structure'] for x in samples.values()]),dynamic_winner=structural([x['dynamic_structure'] for x in samples.values()]),claim='descriptive only, no causal interpretation'))
post=dict(STATUS='PASS',SAMPLE_COMPLETE_N=len(rows),PLANNED_TRAJECTORIES=len(plan),EVALUATED_TRAJECTORIES=len(raw),MISSING_N=0,ERROR_N=0,DUPLICATE_EXECUTION_N=0,UNAUTHORIZED_RETRY_N=0,CHECKPOINT_UNCHANGED=True,FORMAL_DYNAMIC_ARTIFACTS_UNCHANGED=True,RANDOM_PLAN_UNCHANGED=True,EXACT_BUDGET_MATCH=True);put('RANDOM_POSTFLIGHT.json',post);put('RANDOM_SUMMARY.json',s)
summary=['SINGLE_REPLICATE_EQUAL_BUDGET_RANDOM_COMPLETE','[AUTHORITY]',f"FORMAL_DYNAMIC_RUN={P['FORMAL_DYNAMIC_RUN']}",'FORMAL_DYNAMIC_STATUS=COMPLETE','CHECKPOINT_MATCH=YES','EXECUTOR_MATCH=YES','PROTOCOL_MATCH=YES','[COHORT]','FORMAL_SEARCHED_N=1433','DYNAMIC_IMPROVED_N=170',f"COHORT_SHA256={sha(R/'RANDOM_BASELINE_COHORT.tsv')}",'[RANDOM]','RANDOM_REPLICATE_N=1','MASTER_RANDOM_SEED=20260923','SAMPLING=UNIFORM_WITHOUT_REPLACEMENT','CONSTANT_TRAJECTORIES_EXCLUDED=YES','DYNAMIC_TRAJECTORIES_EXCLUDED=NO','PLAN_FROZEN_BEFORE_EXECUTION=YES',f"RANDOM_PLAN_SHA256={P['RANDOM_PLAN_SHA256']}",'[BUDGET]','BUDGET_DEFINITION=UNIQUE_FRESH_EVALUATED_NONCONSTANT_TRAJECTORIES',f"TOTAL_DYNAMIC_NONCONSTANT_BUDGET={P['total_budget']}",f'TOTAL_RANDOM_NONCONSTANT_BUDGET={len(raw)}','EXACT_BUDGET_MATCH=YES','[RESULTS]',*[f'{k}={v}' for k,v in s.items()],'[INTEGRITY]',*[f'{k}={v}' for k,v in post.items()],'[CLAIM_LIMITS]','RANDOM_SEED_UNCERTAINTY_ESTIMATED=NO','FULL_SEARCHED_POPULATION_ESTIMATE=NO','DYNAMIC_IMPROVED_COHORT_ONLY=YES','[FINAL]','STATUS=PASS']
(R/'FINAL_MACHINE_SUMMARY.txt').write_text('\n'.join(summary)+'\n');(R/'RANDOM_FINAL_REPORT.md').write_text('# Single-replicate equal-budget random search\n\nOn the outcome-conditioned Dynamic-improved cohort only, one preregistered equal-budget uniform random-search realization recovered '+f"{s['RANDOM_GAIN_RECOVERY_RATIO']*100:.4f}% of Dynamic gain. Dynamic minus Random was {s['DYNAMIC_MINUS_RANDOM_PP']:.4f} pp; paired task-stratified 95% CI {ci}.\n\n"+'This CI quantifies sample uncertainty conditional on one fixed random realization. It does not estimate seed uncertainty or expected random-search performance. No full-searched-population or universal superiority claim is supported. Task results are descriptive.\n\n```text\n'+'\n'.join(summary)+'\n```\n')
files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and not p.name.endswith('.lock') and p.name not in ['run.log','STATUS.json']);(R/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in files));print('\n'.join(summary))
