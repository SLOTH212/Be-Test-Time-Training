"""CPU-only statistics from packaged V2 evidence. Never imports the model runtime."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,gzip,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def csvrows(p,delim=','):return list(csv.DictReader((ROOT/p).open(),delimiter=delim))
def jsonlines(p):
 f=gzip.open(ROOT/p,'rt') if str(p).endswith('.gz') else (ROOT/p).open()
 with f:return [json.loads(s) for s in f if s.strip()]
def check(a,b):
 if isinstance(b,dict):
  for k,v in b.items():check(a[k],v)
 elif isinstance(b,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):check(x,y)
 elif isinstance(b,(int,float)) and not isinstance(b,bool):assert abs(a-b)<1e-10,(a,b)
 else:assert a==b,(a,b)
def ci(items,seed=20260921,batch=1000):
 rng=np.random.default_rng(seed);sums=np.zeros(20000)
 for task in sorted({t for t,d in items}):
  v=np.array([d for t,d in items if t==task]);n=len(v)
  for k in range(0,20000,batch):sums[k:k+batch]+=v[rng.integers(0,n,size=(batch,n))].sum(1)
 return np.quantile(sums/len(items)*100,[.025,.975],method='linear').tolist()
def mean(v):return float(np.mean(v))*100
def main(out):
 actions=['OFF','L0','L6','L12','L18','L24','ALL'];fixed={};task={}
 for r in csvrows('data/fixed/FIXED7_FULL6500.csv'):
  sid,a=r['sample_id'],r['action'];assert a in actions and a not in fixed.setdefault(sid,{})
  if sid in task:assert task[sid]==r['task']
  task[sid]=r['task'];fixed[sid][a]=float(r['score']);assert 0<=fixed[sid][a]<=1
 assert len(fixed)==6500 and len(set(task.values()))==13 and set(collections.Counter(task.values()).values())=={500}
 assert all(set(v)==set(actions) for v in fixed.values());sb={s:max(v.values()) for s,v in fixed.items()}
 winners=jsonlines('data/dynamic/winners.jsonl');dyn={r['sample_id']:r for r in winners};assert len(dyn)==len(winners)==1433
 assert set(dyn)=={s for s in fixed if task[s]!='ruler_cwe_16k' and sb[s]<1}
 order=jsonlines('data/dynamic/fixed_all6500.jsonl');assert len(order)==len({r['sample_id'] for r in order})==6500
 rs=[]
 for x in order:
  s=x['sample_id'];assert x['sample_best']==sb[s] and x['task']==task[s];d=dyn[s]['dynamic_score'] if s in dyn else sb[s]
  assert d>=sb[s];rs.append(dict(sid=s,task=task[s],sb=sb[s],d=d,delta=d-sb[s]))
 def stat(rr):return dict(N=len(rr),sample_best=mean([r['sb'] for r in rr]),sequential_score=mean([r['d'] for r in rr]),gain_pp=mean([r['delta'] for r in rr]),gain_ci95_pp=ci([(r['task'],r['delta']) for r in rr]),improved_n=sum(r['delta']>1e-12 for r in rr),equal_n=sum(abs(r['delta'])<=1e-12 for r in rr),worse_n=sum(r['delta']< -1e-12 for r in rr))
 r12=[r for r in rs if r['task']!='ruler_cwe_16k'];searched=[r for r in r12 if r['sid'] in dyn];full=stat(rs);parent=stat(r12);sub=stat(searched);authority=read('results/main.json')
 for actual,expected in [(full,authority['full13_caseB']),(parent,authority['main_v2_dynamic']),(sub,authority['main_v2_dynamic']['searched'])]:check(actual,{k:expected[k] for k in actual})
 core={x['sample_id']:x for x in jsonlines('data/mechanism/core.jsonl.gz')};r2={x['sample_id']:x for x in jsonlines('data/mechanism/r2.jsonl.gz')};co=read('experiments/mechanism/COHORT.json');assert set(core)==set(r2)=={r['sid'] for r in r12 if r['delta']>1e-12}
 mech={}
 for label in ['reverse','reset','nontrivial_reset','deletion','sham','K0','K1','r2']:
  pairs=[]
  for c in co:
   s=c['sample_id'];x=core[s];n=c['native_score'];assert n==dyn[s]['dynamic_score'] and c['sample_best']==sb[s]
   if label=='r2':v=r2[s]['r2_score']
   elif label=='reverse':v=x['reverse']['score']
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
    v=x['K'][label]['score']
   pairs.append((c['task'],n,v))
  z=dict(n=len(pairs),native_percent=mean([n for t,n,v in pairs]),condition_percent=mean([v for t,n,v in pairs]),native_minus_condition_pp=mean([n-v for t,n,v in pairs]),CI95=ci([(t,n-v) for t,n,v in pairs]),lower=sum(v<n-1e-12 for t,n,v in pairs),equal=sum(abs(v-n)<=1e-12 for t,n,v in pairs),higher=sum(v>n+1e-12 for t,n,v in pairs));check(z,read('results/mechanism/SUMMARY.json')['results'][label]);mech[label]=z
 rc=0
 for c in co:
  x=r2[c['sample_id']];keys={(r['A'],r['B'],r['tau']) for r in x['candidates']}
  assert len(keys)==len(x['candidates'])==42*(c['T']-1);assert keys=={(a,b,t) for a in actions for b in actions if a!=b for t in range(1,c['T'])};assert max(r['score'] for r in x['candidates'])==x['r2_score'];rc+=len(keys)
 assert rc==97356
 plan=jsonlines('data/random/RANDOM_PLAN.jsonl.gz');raw=jsonlines('data/random/RANDOM_RESULTS.jsonl.gz');assert len(raw)==len(plan)==62772
 planned={(r['sample_id'],r['trajectory_index']):r for r in plan};assert len(planned)==len(plan);seen=set();group=collections.defaultdict(list)
 for r in raw:
  key=r['sample_id'],r['trajectory_index'];assert key not in seen;seen.add(key);p=planned[key];assert r['trajectory']==p['trajectory'] and r['trajectory_hash']==p['trajectory_hash'];group[r['sample_id']].append(r)
 random=[]
 for row in sorted(csvrows('data/random/RANDOM_SAMPLE_SUMMARY.tsv','\t'),key=lambda r:(r['task'],r['sample_id'])):
  s=row['sample_id'];rr=group[s];assert len(rr)==int(row['B_i']);best=max(sb[s],max(r['score'] for r in rr));assert best==float(row['Random']);random.append(dict(task=task[s],sb=sb[s],d=dyn[s]['dynamic_score'],r=best))
 assert len(random)==170
 z=dict(N=len(random),SB_MEAN=mean([r['sb'] for r in random]),DYNAMIC_MEAN=mean([r['d'] for r in random]),RANDOM_MEAN=mean([r['r'] for r in random]),DYNAMIC_MINUS_RANDOM_PP=mean([r['d']-r['r'] for r in random]),RANDOM_GAIN_RECOVERY_RATIO=sum(r['r']-r['sb'] for r in random)/sum(r['d']-r['sb'] for r in random),RANDOM_EQ_DYNAMIC_N=sum(r['r']==r['d'] for r in random),RANDOM_LT_DYNAMIC_N=sum(r['r']<r['d'] for r in random),DYNAMIC_MINUS_RANDOM_CI=ci([(r['task'],r['d']-r['r']) for r in random],20260923,20000))
 expected=read('results/random/RANDOM_SUMMARY.json');check(z,{k:expected[k] for k in z})
 result=dict(status='PASS',numpy=np.__version__,fixed_cells=45500,dynamic12=parent,searched=sub,full13_caseB=full,mechanism=mech,r2_candidates=rc,random=z,GPU_USED=False,MODEL_LOADED=False)
 out.mkdir(parents=True,exist_ok=True);(out/'RECOMPUTED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/recompute');main(p.parse_args().out.resolve())
