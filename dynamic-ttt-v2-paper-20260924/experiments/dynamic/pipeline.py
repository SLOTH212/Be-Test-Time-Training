#!/usr/bin/env python3
"""Portable, sample-locked V2 Dynamic execution. No automatic retries or monitoring."""
import os,sys
# No numerical libraries are imported until the process has its assigned CPU mask.
from pathlib import Path
import argparse,json,hashlib,subprocess,fcntl,datetime,traceback,time,collections,importlib.metadata
R=Path(__file__).resolve().parent
P=json.loads((R/'PROTOCOL.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def get(p):return json.loads(Path(p).read_text())
def key(s):return hashlib.sha256(s.encode()).hexdigest()
def cpus(s):
 out=set()
 for part in s.split(','):
  if '-' in part:a,b=map(int,part.split('-'));out.update(range(a,b+1))
  else:out.add(int(part))
 return out
def affinity(s):
 target=cpus(s)
 if not target<=os.sched_getaffinity(0):raise RuntimeError(f'CPU_SET_NOT_AVAILABLE {s}')
 os.sched_setaffinity(0,target)
def verify():
 manifest=get(R/'MANIFEST.json')
 for name,digest in manifest.items():
  if sha(R/name)!=digest:raise RuntimeError('PACKAGE_HASH_MISMATCH '+name)
 return sha(R/'PROTOCOL.json')
def engine(out):
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 sys.path.insert(0,str(R/'runtime'))
 import torch,types,executor,scorer
 for pkg,version in P['expected_versions'].items():
  if importlib.metadata.version(pkg)!=version:raise RuntimeError('ENVIRONMENT_VERSION_MISMATCH '+pkg)
 torch.set_num_threads(len(os.sched_getaffinity(0)));torch.set_num_interop_threads(1)
 torch.manual_seed(P['seed']);torch.cuda.manual_seed_all(P['seed'])
 ctx=types.SimpleNamespace(code=R/'runtime/code',ckpt=R/'model',run=out,score_fn=scorer.per_sample_score,training_chunk=1024,chunk=1024)
 model,tok=executor.load_model(ctx)
 device={'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)), 'uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cpus':sorted(os.sched_getaffinity(0))}
 return executor,ctx,model,tok,device
def samples():return {x['sample_id']:x for x in (json.loads(l) for l in (R/'data/samples.jsonl').open())}
def gate_reference(ev,refs,ex):
 records={}
 token_hash=hashlib.sha256(ev.enc0.input_ids.numpy().tobytes()).hexdigest()
 for a in P['actions']:
  rec=ev.evaluate([a]*ev.full_chunks);old=refs[a]
  if token_hash!=old['input_token_sha256'] or rec['score']!=old['score'] or rec['prediction_hash']!=old['prediction_hash']:
   raise RuntimeError('FIXED_REFERENCE_MISMATCH '+json.dumps({'sample_id':ev.row['sample_id'],'action':a,'new_score':rec['score'],'old_score':old['score'],'prediction_match':rec['prediction_hash']==old['prediction_hash'],'token_match':token_hash==old['input_token_sha256']}))
  records[a]=rec
 return records
def work(a,bridge=False):
 affinity(a.cpu_set)
 import faulthandler;faulthandler.enable(all_threads=True)
 a.output.mkdir(parents=True,exist_ok=True);protocol_hash=verify()
 ex,ctx,model,tok,device=engine(a.output);rows=samples();refs=get(R/'data/fixed_reference.json')
 ids=get(R/'data/bridge_ids.json') if bridge else get(R/f'data/assignment_{a.machine}.json')
 if bridge and a.limit:ids=ids[:a.limit]
 if not bridge:
  gate=get(a.output/f'bridge_{a.worker}.json')
  assert gate['status']=='PASS' and gate['protocol_hash']==protocol_hash and gate['device']['uuid']==device['uuid'] and gate['n']==len(get(R/'data/bridge_ids.json')),'BRIDGE_REQUIRED_ON_THIS_GPU'
 done=0;sid=None;begin=time.time()
 try:
  for sid in ids:
   dest=a.output/'results'/f'{key(sid)}.json'
   if not bridge and dest.exists():
    prev=get(dest);assert prev['sample_id']==sid and prev['protocol_hash']==protocol_hash and prev['model_sha256']==P['model_sha256'] and prev['fixed_reference_match'] and prev['dynamic_score']>=prev['sample_best'];continue
   (a.output/'locks').mkdir(exist_ok=True)
   with (a.output/'locks'/f'{"bridge_"+str(a.worker)+"_" if bridge else ""}{key(sid)}.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if not bridge and dest.exists():continue
    put(a.output/f'status_{a.worker}.json',{'state':'BRIDGE' if bridge else 'RUNNING','pid':os.getpid(),'sample_id':sid,'completed_by_this_process':done,'device':device,'at':now()})
    ev=ex.Evaluator(ctx,model,tok,rows[sid],'dynamic_old1k_v2',done+1,len(ids),a.output/f'candidate_progress_{a.worker}.json')
    fixed=gate_reference(ev,refs[sid],ex)
    if not bridge:
     best,visited=ex.beam_search(ev,range(ev.full_chunks))
     replay=ev.evaluate(best['sequence'],capture=True,force=True)
     assert replay['prediction_hash']==best['prediction_hash'] and replay['score']==best['score'],'WINNER_REPLAY_MISMATCH'
     sb=max(r['score'] for r in fixed.values());assert replay['score']>=sb,'DYNAMIC_BELOW_FIXED'
     ev.finish()
     result={'sample_id':sid,'task':rows[sid]['task'],'protocol_hash':protocol_hash,'model_sha256':P['model_sha256'],'executor_sha256':P['executor_sha256'],'machine':a.machine,'device':device,'fixed_reference_match':True,'fixed_scores':{x:y['score'] for x,y in fixed.items()},'sample_best':sb,'dynamic_score':replay['score'],'gain':replay['score']-sb,'candidate_count':len(visited),'winner':replay,'candidates':visited,'at':now()}
     put(dest,result)
    else:ev.finish()
    done+=1;del ev
  if bridge:
   put(a.output/f'bridge_{a.worker}.json',{'status':'PASS','n':done,'protocol_hash':protocol_hash,'device':device,'at':now(),'limited_probe':bool(a.limit)})
  else:put(a.output/f'COMPLETE_{a.worker}.json',{'status':'COMPLETE','new_results':done,'elapsed_seconds':time.time()-begin,'at':now()})
 except BaseException:
  put(a.output/'failures'/f'{a.worker}_{time.time_ns()}.json',{'sample_id':sid,'device':device,'traceback':traceback.format_exc(),'at':now()});raise

def supervise(a,bridge=False):
 affinity('8-23');verify();a.output.mkdir(parents=True,exist_ok=True)
 lock=(a.output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpus=a.gpus.split(',');sets=a.cpu_sets.split(';');assert len(gpus)==len(sets)
 masks=[cpus(c) for c in sets]
 assert all(m<=set(range(8,24)) for m in masks) and all(not (masks[i]&masks[j]) for i in range(len(masks)) for j in range(i)),'OVERLAPPING_OR_DISALLOWED_CPUS'
 if not bridge:
  assignment=get(R/f'data/assignment_{a.machine}.json');config={'protocol_hash':sha(R/'PROTOCOL.json'),'machine':a.machine,'assignment_sha256':sha(R/f'data/assignment_{a.machine}.json'),'gpus':gpus,'cpu_sets':sets}
  if (a.output/'RUN_CONFIG.json').exists():assert get(a.output/'RUN_CONFIG.json')==config,'RESUME_CONFIGURATION_CHANGED'
  else:put(a.output/'RUN_CONFIG.json',config)
 children={};logs=[]
 try:
  for w,(gpu,cpu) in enumerate(zip(gpus,sets)):
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1')
   log=(a.output/f'{"bridge" if bridge else "worker"}_{w}_{time.time_ns()}.log').open('wb');logs.append(log)
   cmd=[sys.executable,'-u','-X','faulthandler',str(R/'pipeline.py'),'bridge-worker' if bridge else 'worker','--worker',str(w),'--cpu-set',cpu,'--machine',a.machine,'--output',str(a.output)]
   if bridge and a.limit:cmd+=['--limit',str(a.limit)]
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);children[proc.pid]=proc
  put(a.output/'PROCESS_INFO.json',{'supervisor_pid':os.getpid(),'workers':list(children),'at':now(),'mode':'bridge' if bridge else 'search'})
  while children:
   pid,raw=os.wait();code=os.waitstatus_to_exitcode(raw);children.pop(pid)
   if code:raise RuntimeError(f'WORKER_EXIT pid={pid} code={code}')
  if not bridge:
   missing=[sid for sid in assignment if not (a.output/'results'/f'{key(sid)}.json').exists()]
   assert not missing,f'MISSING_RESULTS {len(missing)}'
   put(a.output/'COMPLETE.json',{'status':'COMPLETE','n':len(assignment),'machine':a.machine,'protocol_hash':sha(R/'PROTOCOL.json'),'at':now()})
 except BaseException:
  for proc in children.values():proc.terminate()
  for proc in children.values():proc.wait()
  put(a.output/'supervisor_failure.json',{'traceback':traceback.format_exc(),'at':now()});raise

def merge(a):
 verify();records={};protocol_hash=sha(R/'PROTOCOL.json')
 for directory in a.inputs:
  for p in (directory/'results').glob('*.json'):
   x=get(p);sid=x['sample_id'];assert sid not in records,'DUPLICATE_SAMPLE '+sid
   assert x['protocol_hash']==protocol_hash and x['model_sha256']==P['model_sha256'] and x['fixed_reference_match'] and x['dynamic_score']>=x['sample_best'];records[sid]=x
 expected=set(get(R/'data/assignment_remote.json')+get(R/'data/assignment_local.json'))
 assert set(records)==expected,f'INCOMPLETE_OR_EXTRA missing={len(expected-set(records))} extra={len(set(records)-expected)}'
 fixed=[json.loads(l) for l in (R/'data/fixed_all6500.jsonl').open()];tasks={}
 for row in fixed:
  if row['task']=='ruler_cwe_16k':continue
  t=row['task'];tasks.setdefault(t,{'n':0,'sb_sum':0.,'dynamic_sum':0.,'searched':0})
  z=tasks[t];z['n']+=1;z['sb_sum']+=row['sample_best'];z['dynamic_sum']+=records[row['sample_id']]['dynamic_score'] if row['sample_id'] in records else row['sample_best'];z['searched']+=row['sample_id'] in records
 for z in tasks.values():z['sample_best_percent']=100*z['sb_sum']/z['n'];z['dynamic_percent']=100*z['dynamic_sum']/z['n'];z['gain_pp']=z['dynamic_percent']-z['sample_best_percent']
 scopes={}
 for name,ts in [('12task',list(tasks))]:
  n=sum(tasks[t]['n'] for t in ts);scopes[name]={'n':n,'searched':sum(tasks[t]['searched'] for t in ts),'sample_best_percent':100*sum(tasks[t]['sb_sum'] for t in ts)/n,'dynamic_percent':100*sum(tasks[t]['dynamic_sum'] for t in ts)/n}
 put(a.report,{'status':'COMPLETE','n_searched':len(records),'protocol_hash':protocol_hash,'scopes':scopes,'tasks':tasks,'at':now()})
def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 sub.add_parser('verify')
 for cmd in ['bridge','run','launch','worker','bridge-worker']:
  q=sub.add_parser(cmd);q.add_argument('--output',type=Path,required=True);q.add_argument('--machine',choices=['remote','local'],default='remote')
  if cmd in ['worker','bridge-worker']:q.add_argument('--worker',type=int,required=True);q.add_argument('--cpu-set',required=True)
  else:q.add_argument('--gpus',default='0,1,2,3');q.add_argument('--cpu-sets',default='8-11;12-15;16-19;20-23')
  if cmd in ['bridge','bridge-worker']:q.add_argument('--limit',type=int,default=0)
 q=sub.add_parser('merge');q.add_argument('--inputs',type=Path,nargs='+',required=True);q.add_argument('--report',type=Path,required=True)
 a=parser.parse_args()
 if hasattr(a,'output'):a.output=a.output.resolve()
 if a.command=='verify':print('VERIFY_PASS',verify())
 elif a.command=='bridge':supervise(a,True)
 elif a.command=='run':supervise(a)
 elif a.command=='worker':work(a)
 elif a.command=='bridge-worker':work(a,True)
 elif a.command=='merge':merge(a)
 else:
  verify();a.output.mkdir(parents=True,exist_ok=True)
  with (a.output/'launch.log').open('ab') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(R/'pipeline.py'),'run','--output',str(a.output),'--machine',a.machine,'--gpus',a.gpus,'--cpu-sets',a.cpu_sets],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  print(json.dumps({'supervisor_pid':proc.pid,'output':str(a.output),'note':'Detached launch requested; inspect PROCESS_INFO/status files for startup. No periodic monitoring.'}))
if __name__=='__main__':main()
