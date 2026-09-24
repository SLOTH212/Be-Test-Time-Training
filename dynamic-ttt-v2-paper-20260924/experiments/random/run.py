import os,sys
W=int(sys.argv[1]) if len(sys.argv)>1 else None
os.sched_setaffinity(0,set(range(8+8*W,16+8*W)) if W is not None else set(range(8,24)))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
from pathlib import Path
import json,csv,types,hashlib,fcntl,subprocess,traceback,collections,importlib.metadata,time
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'runtime'))
import executor as ex
import scorer
P=json.loads((R/'RANDOM_PROTOCOL.json').read_text());PH=ex.sha(R/'RANDOM_PROTOCOL.json');B=Path(P['formal_bundle'])
def put(n,x):ex.atomic_json(R/n,x)
def gate():
 assert json.loads((R/'CPU_PREFLIGHT.json').read_text())['status']=='PASS'
 assert ex.sha(R/'RANDOM_PLAN.jsonl')==P['RANDOM_PLAN_SHA256']
 for n,h in json.loads((R/'FROZEN_FILES.json').read_text()).items():assert ex.sha(R/n)==h,n
 assert ex.sha(R/'runtime/executor.py')==P['EXECUTOR_SHA256']
def structure(seq):return dict(switch_count=sum(a!=b for a,b in zip(seq,seq[1:])),distinct_action_count=len(set(seq)),first_action=seq[0],last_action=seq[-1])
def append(f,x):f.write(json.dumps(x,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def worker():
 gate()
 for pkg,version in P['expected_versions'].items():assert importlib.metadata.version(pkg)==version,(pkg,version)
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(P['executor_rng_seed']);torch.cuda.manual_seed_all(P['executor_rng_seed'])
 assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5090' and torch.cuda.get_device_capability(0)==(12,0)
 ctx=types.SimpleNamespace(code=R/'runtime/code',ckpt=Path(P['model_path']),run=R,score_fn=scorer.per_sample_score,training_chunk=1024,chunk=1024)
 m,tok=ex.load_model(ctx);co=ex.read_csv(R/'RANDOM_BASELINE_COHORT.tsv') if False else list(csv.DictReader((R/'RANDOM_BASELINE_COHORT.tsv').open(),delimiter='\t'))
 rows={x['sample_id']:x for x in ex.load_jsonl(B/'data/samples.jsonl')};ref=json.loads((B/'data/fixed_reference.json').read_text());plan=collections.defaultdict(list)
 for x in ex.load_jsonl(R/'RANDOM_PLAN.jsonl'):plan[x['sample_id']].append(x)
 co=co[W::2];done=0;budget=sum(len(plan[c['sample_id']]) for c in co);started=time.time()
 # Exclusive creation intentionally forbids silent retries or restarting a partial replicate.
 with (R/f'WORKER_RESULTS_{W}.jsonl').open('x') as out,(R/f'EXECUTION_LEDGER_{W}.jsonl').open('x') as ledger:
  for idx,c in enumerate(co,1):
   sid=c['sample_id'];row=rows[sid];ev=ex.Evaluator(ctx,m,tok,row,'random_equal_budget_replicate0',idx,len(co),R/f'parity_progress_{W}.json')
   tokenhash=hashlib.sha256(ev.enc0.input_ids.numpy().tobytes()).hexdigest();fixed={}
   for a in P['ACTION_SET']:
    z=ev.evaluate([a]*ev.full_chunks,force=True);rr=ref[sid][a];assert z['score']==rr['score'] and z['prediction_hash']==rr['prediction_hash'] and tokenhash==rr['input_token_sha256'];fixed[a]=z
   native=ev.evaluate(json.loads(c['dynamic_winner_trajectory']),force=True);assert native['score']==float(c['dynamic_score']) and native['prediction_hash']==c['dynamic_prediction_hash'];ev.finish()
   put(f'parity/{hashlib.sha256(sid.encode()).hexdigest()}.json',dict(status='PASS',sample_id=sid,worker=W,shared_constants=7,dynamic_winner_verification=1,excluded_from_random_budget=True,at=ex.now()))
   sb=float(c['sample_best_score']);assert max(z['score'] for z in fixed.values())==sb
   winner_action=next(a for a in P['ACTION_SET'] if fixed[a]['score']==sb);best=fixed[winner_action];best_index=None;randbest=None;randbest_index=None
   for entry in plan[sid]:
    k=(sid,entry['trajectory_index']);append(ledger,dict(event='START',sample_id=sid,trajectory_index=k[1],trajectory_hash=entry['trajectory_hash'],at=ex.now()))
    z=ev.evaluate(entry['trajectory'],force=True);assert z['sequence_hash']==entry['trajectory_hash'] and z['clean_full_replay'] and z['recompute_downstream_updates'] and not z['stored_delta_stitching'] and z['generation_update_count']==0
    result=dict(**entry,score=z['score'],prediction_hash=z['prediction_hash'],latency_seconds=z['latency_seconds'],checkpoint_sha256=z['checkpoint_hash'],protocol_sha256=PH,worker=W,**structure(entry['trajectory']),clean_full_replay=True,generation_update_count=0,at=ex.now());append(out,result);append(ledger,dict(event='END',sample_id=sid,trajectory_index=k[1],trajectory_hash=entry['trajectory_hash'],at=ex.now()));done+=1
    if randbest is None or z['score']>randbest['score']:randbest=z;randbest_index=k[1]
    if z['score']>best['score']:best=z;best_index=k[1]
    if done%10==0:put(f'STATUS_{W}.json',dict(state='RUNNING',worker=W,cpus=sorted(os.sched_getaffinity(0)),completed_samples=idx-1,total_samples=len(co),completed_random_trajectories=done,total_random_trajectories=budget,current_sample=sid,current_index=k[1],current_best=best['score'],elapsed_seconds=time.time()-started,at=ex.now()))
   ev.finish();assert best['score']>=sb
   put(f'samples/{hashlib.sha256(sid.encode()).hexdigest()}.json',dict(sample_id=sid,task=c['task'],sample_best=sb,dynamic_score=float(c['dynamic_score']),random_score=best['score'],random_gain=best['score']-sb,B_i=len(plan[sid]),winner_source='shared_constant' if best_index is None else 'random',winner_trajectory=best['sequence'],winner_index=best_index,winner_structure=structure(best['sequence']),best_nonconstant_score=randbest['score'],best_nonconstant_trajectory=randbest['sequence'],best_nonconstant_index=randbest_index,best_nonconstant_structure=structure(randbest['sequence']),dynamic_structure=structure(json.loads(c['dynamic_winner_trajectory'])),protocol_sha256=PH,at=ex.now()))
   print(json.dumps(dict(worker=W,samples_done=idx,total=len(co),random_done=done,at=ex.now())),flush=True)
 assert done==budget and ex.sha(Path(P['model_path'])/'model.safetensors')==P['CHECKPOINT_IDENTITY']
 put(f'COMPLETE_{W}.json',dict(status='COMPLETE',samples=len(co),random_trajectories=done,at=ex.now()));put(f'STATUS_{W}.json',dict(state='COMPLETE',completed_samples=len(co),completed_random_trajectories=done,at=ex.now()))
if __name__=='__main__':
 try:
  if W is not None:worker()
  else:
   lock=(R/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate();put('STATUS.json',dict(state='RUNNING',replicate=0,at=ex.now()))
   children={}
   for w in range(2):
    proc=subprocess.Popen([sys.executable,'-u',str(R/'run.py'),str(w)]);children[proc.pid]=proc
   put('PROCESS_INFO.json',dict(supervisor=os.getpid(),workers=list(children),cpu_sets=['8-15','16-23'],at=ex.now()))
   while children:
    pid,raw=os.wait();code=os.waitstatus_to_exitcode(raw);children.pop(pid)
    if code:
     for proc in children.values():proc.terminate()
     for proc in children.values():proc.wait()
     raise RuntimeError(f'WORKER_FAILED {pid} {code}; no automatic retry')
   subprocess.run([sys.executable,str(R/'finalize.py')],check=True)
   put('STATUS.json',dict(state='COMPLETE',at=ex.now()))
 except BaseException:
  put('FAILURE.json' if W is None else f'FAILURE_{W}.json',dict(traceback=traceback.format_exc(),at=ex.now()))
  if W is None:put('STATUS.json',dict(state='FAILED',at=ex.now()))
  raise
