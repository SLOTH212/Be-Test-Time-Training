import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;C=json.loads((R/'COHORT.json').read_text());core={x['sample_id']:x for x in (json.loads(p.read_text()) for p in (R/'core').glob('*.json'))};r2={x['sample_id']:x for x in (json.loads(p.read_text()) for p in (R/'r2').glob('*.json'))}
def stat(items):
 # tuple task,native,condition
 if not items:return dict(n=0)
 a=np.array([x[1] for x in items]);b=np.array([x[2] for x in items]);d=a-b;rng=np.random.default_rng(20260921);s=np.zeros(20000)
 for t in sorted({x[0] for x in items}):
  v=np.array([d[i] for i,x in enumerate(items) if x[0]==t]);n=len(v)
  for j in range(0,20000,1000):s[j:j+1000]+=v[rng.integers(0,n,(1000,n))].sum(1)
 return dict(n=len(items),native_percent=float(a.mean()*100),condition_percent=float(b.mean()*100),native_minus_condition_pp=float(d.mean()*100),CI95=np.quantile(s/len(items)*100,[.025,.975]).tolist(),lower=int((b<a-1e-12).sum()),equal=int((abs(b-a)<=1e-12).sum()),higher=int((b>a+1e-12).sum()))
report=dict(core_completed=len(core),r2_completed=len(r2),parent=170,P='omitted; V2 all-five materialization invariant',bootstrap='20000 paired task-stratified replicates, seed20260921',results={})
for label in ['reverse','reset','nontrivial_reset','deletion','sham','K0','K1','K1_minus_K0','r2']:
 items=[]
 for c in C:
  x=core.get(c['sample_id']);task=c['task'];n=c['native_score']
  if label=='r2':
   if c['sample_id'] in r2:items.append((task,n,r2[c['sample_id']]['r2_score']))
   continue
  if not x:continue
  if label=='reverse':v=x['reverse']['score']
  elif label in ['reset','nontrivial_reset']:
   if not c['reset_eligible'] or (label=='nontrivial_reset' and not c['nontrivial_reset']):continue
   v=x['reset']['score']
  elif label=='deletion':
   if not x['deletion']:continue
   v=sum(z['result']['score'] for z in x['deletion'])/len(x['deletion'])
  elif label=='sham':
   if not x['sham']:continue
   v=x['sham']['score']
  else:
   if not x['K']:continue
   if label=='K1_minus_K0':n=x['K']['K1']['score'];v=x['K']['K0']['score']
   else:v=x['K'][label]['score']
  items.append((task,n,v))
 report['results'][label]=stat(items)
ks=[x['K'] for x in core.values() if x['K']];report['K_closure']=dict(n=len(ks),K0_trace_exact=sum(x['K0_trace_closure'] for x in ks),K1_trace_exact=sum(x['K1_trace_closure'] for x in ks),K0_prediction_exact=sum(x['K0_prediction_closure'] for x in ks))
if r2:
 vals=list(r2.values());report['R2']=dict(n=len(vals),candidates=sum(x['candidate_count'] for x in vals),reused=sum(x['reused'] for x in vals),at_least_native=sum(x['matches_native'] for x in vals),exact_native=sum(abs(x['r2_score']-x['native_score'])<1e-12 for x in vals),native_one_switch=sum(x['native_switches']==1 for x in vals),native_multi_switch=sum(x['native_switches']>1 for x in vals),multi_switch_at_least_native=sum(x['native_switches']>1 and x['matches_native'] for x in vals))
(R/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');lines=['# V2 mechanism replication (P omitted)','',f'Core complete: {len(core)}/170; R2 complete: {len(r2)}/170.','', '| Experiment | N | Native % | Condition % | Drop pp | 95% CI |','|---|---:|---:|---:|---:|---|']
for k,x in report['results'].items():
 if x['n']:lines.append(f"| {k} | {x['n']} | {x['native_percent']:.4f} | {x['condition_percent']:.4f} | {x['native_minus_condition_pp']:.4f} | {x['CI95']} |")
lines+=['','Selection is conditional on current Dynamic improvement; sample sizes differ from historical cohorts. R2 is hindsight reference-scored exhaustive one-switch search. K closure uses explicitly recorded trace hashes, not an unmeasured claim of full historical L1-L6 coverage.','',json.dumps(report.get('K_closure'),indent=2),json.dumps(report.get('R2'),indent=2)]
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
