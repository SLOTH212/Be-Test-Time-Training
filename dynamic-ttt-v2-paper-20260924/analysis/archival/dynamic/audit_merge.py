import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
R=Path(__file__).resolve().parent;B=Path('/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918');T=R/'dynamic_old1k_v2_remote_transfer_20260921'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
checked=0
for line in (T/'BUNDLE_SHA256SUMS.txt').read_text().splitlines():
 if not line.strip():continue
 h,n=line.split(maxsplit=1);n=n.lstrip('*');p=(T/n).resolve();assert p.is_relative_to(T.resolve());assert sha(p)==h,n;checked+=1
assert sha(T/'provenance/PROTOCOL.json')==sha(B/'PROTOCOL.json')
P=json.loads((B/'PROTOCOL.json').read_text());ph=sha(B/'PROTOCOL.json');refs=json.loads((B/'data/fixed_reference.json').read_text());records={};sources={}
for machine,root in [('local',Path('/home/zonghan/ttt/runs/formal/dynamic_old1k_v2_noncwe_local_20260918')),('remote',T/'dynamic_old1k_v2_remote_results')]:
 expected=set(json.loads((B/f'data/assignment_{machine}.json').read_text()));got=set()
 for p in (root/'results').glob('*.json'):
  x=json.loads(p.read_text());sid=x['sample_id'];assert sid not in records and sid in expected;assert x['protocol_hash']==ph and x['model_sha256']==P['model_sha256'] and x['executor_sha256']==P['executor_sha256'];assert x['fixed_reference_match'];assert x['fixed_scores']=={a:v['score'] for a,v in refs[sid].items()}
  sb=max(x['fixed_scores'].values());d=x['dynamic_score'];assert abs(sb-x['sample_best'])<1e-12 and abs(d-sb-x['gain'])<1e-12 and d>=sb
  w=x['winner'];assert w['score']==d and w['ttt_chunk_size']==1024 and w['checkpoint_hash']==P['model_sha256'];assert w['generation_update_count']==0 and w['clean_full_replay'] and w['recompute_downstream_updates'] and not w['stored_delta_stitching']
  assert len(x['candidates'])==x['candidate_count'] and max(z['score'] for z in x['candidates'])==d
  assert any(z['sequence']==w['sequence'] and z['prediction_hash']==w['prediction_hash'] for z in x['candidates'])
  records[sid]=x;got.add(sid);sources[sid]={'path':str(p),'sha256':sha(p)}
 assert got==expected
fixed=[json.loads(l) for l in (B/'data/fixed_all6500.jsonl').read_text().splitlines() if l.strip()];fixed=[x for x in fixed if x['task']!='ruler_cwe_16k'];assert len(fixed)==6000
assert set(records)=={x['sample_id'] for x in fixed if x['sample_best']<1}
rows=[]
for f in fixed:
 sid=f['sample_id'];x=records.get(sid);sb=f['sample_best'];d=x['dynamic_score'] if x else sb
 rows.append(dict(sample_id=sid,task=f['task'],sample_best=sb,dynamic_score=d,gain=d-sb,searched=x is not None,machine=x['machine'] if x else 'ceiling',candidate_count=x['candidate_count'] if x else 0))
def summarize(rs):
 n=len(rs);return dict(n=n,sample_best=sum(x['sample_best'] for x in rs)/n*100,dynamic=sum(x['dynamic_score'] for x in rs)/n*100,gain_pp=sum(x['gain'] for x in rs)/n*100,improved=sum(x['gain']>1e-12 for x in rs),equal=sum(abs(x['gain'])<=1e-12 for x in rs),worse=sum(x['gain']< -1e-12 for x in rs))
def ci(rs):
 rng=np.random.default_rng(20260921);sums=np.zeros(20000)
 for task in sorted({x['task'] for x in rs}):
  vals=np.array([x['gain'] for x in rs if x['task']==task]);n=len(vals)
  for i in range(0,20000,1000):sums[i:i+1000]+=vals[rng.integers(0,n,size=(1000,n))].sum(axis=1)
 return np.quantile(sums/len(rs)*100,[.025,.975]).tolist()
searched=[x for x in rows if x['searched']];report=dict(status='PASS',transfer_files_verified=checked,protocol_sha256=ph,full6000=summarize(rows),searched1433=summarize(searched),per_task={t:summarize([x for x in rows if x['task']==t]) for t in sorted({x['task'] for x in rows})},bootstrap=dict(seed=20260921,replicates=20000,method='paired task-stratified percentile, sample-weighted; separately resample each reported scope',full6000_gain_CI=ci(rows),searched1433_gain_CI=ci(searched)))
oldroot=Path('/home/zonghan/ttt/runs/formal/dynamic_nonceiling_12task_1p7b_v1/run_20260828T161330+0800');oldfixed={x['sample_id']:x for x in csv.DictReader((oldroot/'eligibility/fixed7_sample_best_full6500.csv').open())};oldres={x['sample_id']:x for x in (json.loads(p.read_text()) for p in (oldroot/'results').glob('*.json'))};oldrows=[]
for x in rows:
 sid=x['sample_id'];sb=float(oldfixed[sid]['sample_best']);d=oldres[sid]['dynamic_score'] if sid in oldres else sb;oldrows.append(dict(sample_id=sid,sample_best=sb,dynamic_score=d,gain=d-sb))
a={x['sample_id'] for x in oldrows if x['gain']>1e-12};b={x['sample_id'] for x in rows if x['gain']>1e-12};report['historical_full6000']=summarize(oldrows);report['improved_sets']=dict(both=len(a&b),old_only=len(a-b),new_only=len(b-a),old_only_ids=sorted(a-b),new_only_ids=sorted(b-a))
for name,data in [('FINAL_AUDIT.json',report),('RESULT_SOURCES.json',sources)]: (R/name).write_text(json.dumps(data,indent=2)+'\n')
with (R/'FINAL_PER_SAMPLE_6000.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(report,indent=2))
