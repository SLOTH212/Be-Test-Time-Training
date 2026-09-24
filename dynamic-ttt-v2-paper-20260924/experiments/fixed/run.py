import os,sys
W=int(sys.argv[1]) if len(sys.argv)>1 else None
CPUS=set(range(8+8*W,16+8*W)) if W is not None else set(range(8,24))
os.sched_setaffinity(0,CPUS)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
from pathlib import Path
import json,csv,hashlib,types,fcntl,subprocess,traceback,time,collections
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
import executor as ex
import scorer
P=json.loads((R/'PROTOCOL.json').read_text());PH=ex.sha(R/'PROTOCOL.json')
ex.CHECKPOINT_HASH=P['model_sha256']
def preflight():
 assert ex.sha(Path(P['benchmark']))==P['benchmark_sha256']
 assert ex.sha(R/'reused/V2_FIXED7_1435.csv')==P['reused_sha256']
 assert ex.sha(R/'run.py')==P['runner_sha256']
 for f,h in P['source_hashes'].items():assert ex.sha(R/f)==h,f
 rows=ex.load_jsonl(P['benchmark']);byid={x['sample_id']:x for x in rows};old=ex.read_csv(R/'reused/V2_FIXED7_1435.csv');reuse={}
 assert len(rows)==len(byid)==6500
 for x in old:
  sid=x['sample_id'];a=x['action'];row=byid[sid]
  assert (sid,a) not in reuse and a in ex.ACTIONS
  assert x['status']=='SUCCESS' and x['checkpoint_unchanged']=='True' and x['state_reset_pass']=='True'
  assert x['generation_update_count']==x['tail_update_count']=='0'
  assert hashlib.sha256(row['input'].encode()).hexdigest()==x['input_sha256']
  assert ex.canonical(row['reference'])==x['reference_sha256'] and row['max_new_tokens']==int(x['generation_budget'])
  reuse[sid,a]=x
 ids={sid for sid,a in reuse};assert len(reuse)==10045 and len(ids)==1435
 assert all((sid,a) in reuse for sid in ids for a in ex.ACTIONS)
 missing=[(i,x) for i,x in enumerate(rows) if x['sample_id'] not in ids];assert len(missing)==5065
 return rows,reuse,missing

def worker():
 lock=(R/f'worker_{W}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(P['seed']);torch.cuda.manual_seed_all(P['seed'])
 rows,reuse,missing=preflight();assigned=missing[W::2];done=0;started=time.time();current=None
 def status(state):ex.atomic_json(R/f'STATUS_{W}.json',dict(state=state,worker=W,pid=os.getpid(),cpus=sorted(CPUS),completed_cells=done,total_cells=len(assigned)*7,current_sample=current,elapsed_seconds=time.time()-started,at=ex.now()))
 status('PREFLIGHT')
 ctx=types.SimpleNamespace(code=R/'source/code',ckpt=Path(P['model_path']),run=R,score_fn=scorer.per_sample_score,training_chunk=1024,chunk=1024)
 model,tok=ex.load_model(ctx)
 # Validate every reused sample against the current tokenizer without GPU inference.
 for sid,a in reuse:
  if a!='OFF':continue
  ref=reuse[sid,a];row=next(x for x in rows if x['sample_id']==sid)
  enc=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False)
  assert hashlib.sha256(enc.input_ids.numpy().tobytes()).hexdigest()==ref['input_token_sha256']
  assert hashlib.sha256(enc.attention_mask.numpy().tobytes()).hexdigest()==ref['attention_mask_sha256']
  assert enc.input_ids.shape[1]//1024==int(ref['T'])
 # Each worker reproduces all seven reference samples/actions under its own CPU affinity.
 for ordinal,sid in enumerate(P['bridge_ids'],1):
  row=next(x for x in rows if x['sample_id']==sid)
  ev=ex.Evaluator(ctx,model,tok,row,'fixed1k_bridge',ordinal,7,R/f'candidate_{W}.json')
  for a in ex.ACTIONS:
   z=ev.evaluate([a]*ev.full_chunks);ref=reuse[sid,a]
   assert abs(z['score']-float(ref['score']))<1e-12 and z['prediction_hash']==ref['prediction_hash'],(sid,a)
  ev.finish()
 ex.atomic_json(R/f'BRIDGE_{W}.json',dict(status='PASS',cells=49,score_prediction_exact=True,reused_token_identity_all1435=True,cpus=sorted(CPUS),at=ex.now()))
 for ordinal,(idx,row) in enumerate(assigned,1):
  current=row['sample_id'];target=R/'new_samples'/f'{idx:06d}.json'
  if target.exists():
   x=json.loads(target.read_text());assert x['protocol_sha256']==PH and x['sample_id']==current and set(x['fixed'])==set(ex.ACTIONS)
   done+=7;continue
  status('RUNNING');ev=ex.Evaluator(ctx,model,tok,row,'fixed7_old1k_v2',ordinal,len(assigned),R/f'candidate_{W}.json');fixed={}
  for a in ex.ACTIONS:
   z=ev.evaluate([a]*ev.full_chunks,capture=False)
   assert z['checkpoint_hash']==P['model_sha256'] and z['ttt_chunk_size']==1024 and 0<=z['score']<=1
   fixed[a]=z
  ev.finish()
  ex.atomic_json(target,dict(sample_id=current,task=row['task'],sample_index=idx,protocol_sha256=PH,model_sha256=P['model_sha256'],input_sha256=hashlib.sha256(row['input'].encode()).hexdigest(),input_token_sha256=hashlib.sha256(ev.enc0.input_ids.numpy().tobytes()).hexdigest(),reference_sha256=ex.canonical(row['reference']),generation_budget=row['max_new_tokens'],fixed=fixed,checkpoint_unchanged=True,worker=W,at=ex.now()))
  done+=7;status('RUNNING')
  if ordinal%25==0:print(json.dumps(dict(worker=W,samples=ordinal,total=len(assigned),at=ex.now())),flush=True)
 assert done==len(assigned)*7;status('COMPLETE');ex.atomic_json(R/f'COMPLETE_{W}.json',dict(cells=done,status='COMPLETE',at=ex.now()))

def merge():
 rows,reuse,missing=preflight();cells=[];summaries=[]
 for idx,row in enumerate(rows):
  sid=row['sample_id'];fixed={}
  if (sid,'OFF') in reuse:
   for a in ex.ACTIONS:
    x=reuse[sid,a];fixed[a]=float(x['score']);cells.append(dict(sample_id=sid,task=row['task'],action=a,score=float(x['score']),prediction_hash=x['prediction_hash'],input_token_sha256=x['input_token_sha256'],source='reused_v2_1435'))
  else:
   x=json.loads((R/'new_samples'/f'{idx:06d}.json').read_text());assert x['protocol_sha256']==PH and x['sample_id']==sid and set(x['fixed'])==set(ex.ACTIONS)
   for a in ex.ACTIONS:
    z=x['fixed'][a];fixed[a]=z['score'];cells.append(dict(sample_id=sid,task=row['task'],action=a,score=z['score'],prediction_hash=z['prediction_hash'],input_token_sha256=x['input_token_sha256'],source='new_missing_v2'))
  sb=max(fixed.values());summaries.append(dict(sample_id=sid,task=row['task'],**fixed,sample_best=sb,dynamic_eligible=sb<1))
 assert len(cells)==45500 and len({(x['sample_id'],x['action']) for x in cells})==45500
 ex.write_csv(R/'FIXED7_FULL6500.csv',cells);ex.write_csv(R/'SAMPLE_BEST_FULL6500.csv',summaries)
 ex.atomic_json(R/'DYNAMIC_ELIGIBLE_IDS.json',[x['sample_id'] for x in summaries if x['dynamic_eligible']])
 ex.atomic_json(R/'SUMMARY.json',dict(total_cells=len(cells),reused_cells=10045,new_cells=35455,eligible=sum(x['dynamic_eligible'] for x in summaries),eligible_by_task=dict(collections.Counter(x['task'] for x in summaries if x['dynamic_eligible'])),mean_scores={a:sum(x[a] for x in summaries)/6500*100 for a in ex.ACTIONS+['sample_best']},at=ex.now()))
 ex.atomic_json(R/'COMPLETE.json',dict(state='COMPLETE',total_cells=45500,at=ex.now()))
if __name__=='__main__':
 try:
  if W is not None:worker()
  else:
   lock=(R/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);preflight()
   ex.atomic_json(R/'STATUS.json',dict(state='RUNNING',workers=2,reused_cells=10045,missing_cells=35455,at=ex.now()))
   children=[subprocess.Popen([sys.executable,'-u',str(R/'run.py'),str(w)],cwd=R) for w in range(2)]
   ex.atomic_json(R/'PROCESS_INFO.json',dict(supervisor=os.getpid(),workers=[p.pid for p in children],cpu_sets=['8-15','16-23'],at=ex.now()))
   codes=[p.wait() for p in children]
   assert codes==[0,0],codes
   merge();ex.atomic_json(R/'STATUS.json',dict(state='COMPLETE',total_cells=45500,at=ex.now()))
 except BaseException:
  ex.atomic_json(R/('FAILURE.json' if W is None else f'FAILURE_{W}.json'),dict(error=traceback.format_exc(),at=ex.now()));raise
