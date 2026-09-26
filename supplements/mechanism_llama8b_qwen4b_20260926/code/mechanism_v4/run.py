"""Hash-bound eight-GPU resumption and independently committed mechanism tasks."""
from pathlib import Path
import os,sys,json,time,math,fcntl,subprocess,signal,traceback,argparse,shutil
import engine as e
from supplement import supplement_boundary, selective_sample, selective_plan
R=e.R
S=R/'work/mechanism_v4_8gpu_20260926'
O=R/'runs/mechanism_v4_8gpu_20260926'
OLD_S=R/'work/mechanism_v2_7gpu_20260926'
OLD_O=R/'runs/mechanism_v2_7gpu_20260926'
GPUS=list(range(8))
read,sha,atomic,canonical=e.read,e.sha,e.atomic,e.canonical

def verify(full=False):
 j=read(S/'JOB.json')
 for p,h in j['bindings'].items():
  if full or not str(p).endswith('.safetensors'): assert sha(p)==h,('INPUT_CHANGED',p)
 return j

def task_list(c):
 tasks=[]
 for x in c['cohort']:
  sid=x['sample_id'];stem=sid.replace(':','_');n=len(x['sequence'])
  tasks.append(dict(id=stem+'__baseline',kind='baseline',sample_id=sid))
  tasks.append(dict(id=stem+'__selective',kind='selective',sample_id=sid,plan=selective_plan(x)))
  for start in range(0,len(e.candidates(n)),32):tasks.append(dict(id=stem+'__r'+str(start),kind='r2',sample_id=sid,start=start,end=min(start+32,len(e.candidates(n)))))
 return tasks

def validate_block(x,rs,start=0,end=None):
 cs=e.candidates(len(x['sequence']));end=len(cs) if end is None else end
 assert 0<=start<=end<=len(cs) and len(rs)<=end-start,('R2_BLOCK_LENGTH',x['sample_id'])
 for a,b in zip(cs[start:end],rs):
  assert all(a[k]==b[k] for k in a),('R2_IDENTITY',x['sample_id'])
  assert math.isfinite(b['score']) and 0<=b['score']<=1
  assert len(b['prediction_hash'])==64 and all(c in '0123456789abcdef' for c in b['prediction_hash'])
  assert b['generation_update_count']==0 and b['kv_replaced_layers']==[]
  assert b['observed_update_calls']==sum(len(e.MAPPING[v]) for v in b['sequence'])
  assert b['layer_update_counts']=={str(l):sum(l in e.MAPPING[v] for v in b['sequence']) for l in e.LAYERS}
 return rs

def validate_records(x,d):
 cs=e.candidates(len(x['sequence']));rs=d['records']
 assert d['completed']==len(rs)<=len(cs) and d['total']==len(cs)
 return validate_block(x,rs)

def best_summary(rs):
 best=max(z['score'] for z in rs);tied=[z for z in rs if abs(z['score']-best)<=1e-12]
 win=min(tied,key=lambda z:(z['tau'],e.ACTIONS.index(z['A']),e.ACTIONS.index(z['B']),z['full_schedule_hash']))
 return dict(candidate_count=len(rs),best_score=best,winner=win,ties=len(tied))

def prepare():
 assert not (S/'JOB.json').exists()
 old=e.verify(full=True)
 j=dict(schema='MECHANISM_V4_ORIGINAL_KV_HISTORICAL_SELECTIVE_8GPU',models=old['models'],bindings=dict(old['bindings']),gpus=GPUS,
  predecessor_job=str(OLD_S/'JOB.json'),predecessor_job_sha256=sha(OLD_S/'JOB.json'),
  kv_protocol='USER_ROLLBACK: original single terminal-run-start last-switch boundary; preserve original30/69 eligible cohort, exact original Engine.mechanisms; import verified completed original KV without rerun; no P factor',
  selective_protocol='Migrated1.7B historical deletion_plan adapted to six candidate layers and existing 4K reset eligibility: last-switch boundary; ALL prefix=>six leave-one-layer-out trials, otherwise first nonOFF prefix action layer; lowest never-written layer SHAM when available; preserve native prefix KV and future actions; no filtering on Reset score',
  experiments=['native','reverse','reset','kv','selective_cumulative_layer_deletion','r2'],r2=old['r2'],task_block_size=32)
 for name in ['run.py','engine.py','trace_support.py','supplement.py','PROTOCOL.md']:j['bindings'][str(S/name)]=sha(S/name)
 j['bindings'][str(OLD_S/'JOB.json')]=sha(OLD_S/'JOB.json')
 reference=R/'releases/ttt_mechanism_runtime_v1'
 for name in ['runtime.py','provenance/original/deletion/screening/manifests/SELECTIVE_DELETION_RULE_V1.json','provenance/original/deletion/scripts/selective_worker.py']:
  j['bindings'][str(reference/name)]=sha(reference/name)
 for c in j['models'].values():c['tasks']=task_list(c)
 atomic(S/'JOB.json',j)
 atomic(O/'CPU_PREFLIGHT.json',dict(status='PASS',job_sha256=sha(S/'JOB.json'),models={k:dict(samples=len(c['cohort']),kv_samples=sum(x['reset_screen']['eligible'] for x in c['cohort']),tasks=len(c['tasks'])) for k,c in j['models'].items()}))

def baseline(engine,x):
 return engine.mechanisms(x)

def receipt_path(label,t):return O/label/'tasks'/(t['id']+'.json')
def commit(label,t,result,origin='NEW'):
 atomic(receipt_path(label,t),dict(status='COMMITTED',task=t,result=result,origin=origin,job_sha256=sha(S/'JOB.json'),time=time.time()))

def import_snapshot():
 j=verify();assert not (O/'IMPORT_MANIFEST.json').exists()
 migration=read(O/'STOP_RECEIPT.json');assert migration['all_old_processes_gone']
 for pid in migration['old_pids']:assert not Path('/proc') .joinpath(str(pid)).exists(),('OLD_PROCESS_STILL_PRESENT',pid)
 manifest={};summary={}
 for label,c in j['models'].items():
  total=finals=mechs=0
  for x in c['cohort']:
   stem=x['sample_id'].replace(':','_');source=OLD_O/label/'samples'/stem;dest=O/'imported'/label/stem
   assert not dest.exists(),('IMPORT_DEST_ALREADY_EXISTS',str(dest))
   dest.mkdir(parents=True)
   for name in ['MECHANISMS.json','R2_PROGRESS.json','FINAL.json']:
    f=source/name
    if f.exists():
     h=sha(f);shutil.copyfile(f,dest/name);assert sha(dest/name)==h==sha(f)
     manifest[str(dest/name)]=dict(source=str(f),sha256=h)
   rs=[]
   if (dest/'R2_PROGRESS.json').exists():rs=validate_records(x,read(dest/'R2_PROGRESS.json'));total+=len(rs)
   if (dest/'FINAL.json').exists():
    f=read(dest/'FINAL.json');assert f['status']=='COMMITTED' and f['sample_id']==x['sample_id'] and f['parent_dynamic']==x and f['base_unchanged'] and f['job_sha256']==j['predecessor_job_sha256']
    assert len(rs)==len(e.candidates(len(x['sequence']))) and f['r2']['all_candidates_sha256']==sha(dest/'R2_PROGRESS.json')
    assert all(f['r2'][k]==v for k,v in best_summary(rs).items());finals+=1
   if (dest/'MECHANISMS.json').exists():
    m=read(dest/'MECHANISMS.json');assert (m['native']['score'],m['native']['prediction_hash'])==(x['score'],x['prediction_hash'])
    assert all(m['native_repeat'][k]==m['native'][k] for k in ['score','prediction_hash','final_hashes','suffix_trace'])
    t=next(t for t in c['tasks'] if t['sample_id']==x['sample_id'] and t['kind']=='baseline')
    commit(label,t,m,'VERIFIED_PREDECESSOR');mechs+=1
   for t in c['tasks']:
    if t['sample_id']==x['sample_id'] and t['kind']=='r2' and t['end']<=len(rs):commit(label,t,dict(records=rs[t['start']:t['end']]),'VERIFIED_PREDECESSOR')
  summary[label]=dict(r2_records=total,legacy_finals=finals,baseline_imports=mechs,new_boundary_imports=0)
 atomic(O/'IMPORT_MANIFEST.json',dict(status='PASS',files=manifest,summary=summary,predecessor_job_sha256=j['predecessor_job_sha256'],job_sha256=sha(S/'JOB.json')))

def verify_import():
 m=read(O/'IMPORT_MANIFEST.json');assert m['status']=='PASS' and m['job_sha256']==sha(S/'JOB.json')
 for p,v in m['files'].items():assert sha(p)==v['sha256'],('IMPORT_CHANGED',p)
 return m

def worker(label,gpu,smoke=False):
 j=verify();c=j['models'][label];assert int(os.environ['CUDA_VISIBLE_DEVICES'])==gpu
 engine=e.Engine(c);root=O/label
 atomic(root/f'worker{gpu}_loaded.json',dict(pid=os.getpid(),gpu=gpu,base_hash=engine.base_hash,smoke=smoke,time=time.time()))
 if smoke:
  x=max((x for x in c['cohort'] if x['reset_screen']['eligible']),key=lambda x:len(set(x['sequence'])))
  rs={}
  atomic(root/'smoke_progress.json',dict(sample=x['sample_id'],phase='historical_selective_all_prefix',time=time.time()))
  rs['selective']=selective_sample(engine,x,e.getrow(x['sample_id']))
  sham_x=next(z for z in c['cohort'] if selective_plan(z)['eligible'] and selective_plan(z)['sham_layer'] is not None)
  atomic(root/'smoke_progress.json',dict(sample=sham_x['sample_id'],phase='historical_selective_sham',time=time.time()))
  rs['selective_sham']=selective_sample(engine,sham_x,e.getrow(sham_x['sample_id']))
  # New wrapper equals the unmodified V2 forward on native and R2 schedules.
  row=e.getrow(x['sample_id']);probes=[x['sequence']]+[z['sequence'] for z in e.candidates(len(x['sequence']))[::max(1,len(e.candidates(len(x['sequence'])))//3)]]
  original=[];engine.install(original=True)
  for seq in probes:original.append(engine.run(row,seq,capture=True)[0])
  engine.install()
  for seq,old in zip(probes,original):
   new=engine.run(row,seq,capture=True)[0];assert all(old[k]==new[k] for k in ['score','prediction_hash','final_hashes'])
  assert e.param_hash(engine.model)==engine.base_hash
  atomic(root/'SMOKE.json',dict(status='PASS',job_sha256=sha(S/'JOB.json'),sample=x['sample_id'],gpu=gpu,results=rs,original_v2_parity='PASS',base_unchanged=True))
  return
 verify_import()
 for t in c['tasks']:
  target=receipt_path(label,t)
  if target.exists():continue
  lockpath=root/'locks'/(t['id']+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
  with lockpath.open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if target.exists():continue
   x=next(x for x in c['cohort'] if x['sample_id']==t['sample_id']);row=e.getrow(x['sample_id'])
   def progress(**kw):atomic(root/f'worker{gpu}_progress.json',dict(task=t,gpu=gpu,pid=os.getpid(),time=time.time(),**kw))
   progress(phase=t['kind'])
   if t['kind']=='baseline':result=baseline(engine,x)
   elif t['kind']=='boundary':result=supplement_boundary(engine,x,row,t['tau'],t['kv'])
   elif t['kind']=='selective':
    assert t['plan']==selective_plan(x)
    result=selective_sample(engine,x,row)
   else:
    seed=O/'imported'/label/x['sample_id'].replace(':','_')/'R2_PROGRESS.json'
    old=read(seed)['records'] if seed.exists() else []
    checkpoint=root/'partial'/(t['id']+'.json')
    if checkpoint.exists():
     saved=read(checkpoint);assert saved['task']==t and saved['job_sha256']==sha(S/'JOB.json'),'PARTIAL_IDENTITY'
     rs=saved['records']
    else:rs=old[t['start']:min(t['end'],len(old))]
    validate_block(x,rs,t['start'],t['end'])
    cs=e.candidates(len(x['sequence']))
    for pos,rec in enumerate(rs,t['start']):assert all(rec[k]==v for k,v in cs[pos].items())
    engine.torch.cuda.empty_cache()
    for i in range(t['start']+len(rs),t['end']):
     rec={**cs[i],**engine.run(row,cs[i]['sequence'])[0]};rs.append(rec)
     if len(rs)%8==0 or i+1==t['end']:
      atomic(checkpoint,dict(task=t,records=rs,job_sha256=sha(S/'JOB.json')));progress(phase='r2',completed=len(rs),total=t['end']-t['start'],seconds=rec['seconds'])
    validate_block(x,rs,t['start'],t['end']);assert len(rs)==t['end']-t['start']
    result=dict(records=rs)
   assert e.param_hash(engine.model)==engine.base_hash
   commit(label,t,result);progress(phase='committed')
 atomic(root/f'worker{gpu}_done.json',dict(status='PASS',gpu=gpu,time=time.time()))

def aggregate(label,c):
 results=[]
 for x in c['cohort']:
  ts=[t for t in c['tasks'] if t['sample_id']==x['sample_id']];rs=[];boundaries={};base=None;selective=None
  for t in ts:
   rec=read(receipt_path(label,t));assert rec['status']=='COMMITTED' and rec['task']==t and rec['job_sha256']==sha(S/'JOB.json')
   if t['kind']=='r2':rs.extend(rec['result']['records'])
   elif t['kind']=='boundary':boundaries[str(t['tau'])]=rec['result']
   elif t['kind']=='selective':selective=rec['result']
   else:base=rec['result']
  validate_records(x,dict(records=rs,completed=len(rs),total=len(e.candidates(len(x['sequence'])))))
  assert len(rs)==len(e.candidates(len(x['sequence']))) and len(boundaries)==0 and base and selective
  out=dict(status='COMMITTED',sample_id=x['sample_id'],parent_dynamic=x,baseline=base,boundaries=boundaries,selective=selective,r2=best_summary(rs),job_sha256=sha(S/'JOB.json'))
  atomic(O/label/'samples'/x['sample_id'].replace(':','_')/'FINAL.json',out);results.append(out)
 atomic(O/label/'AGGREGATE.json',dict(samples=len(results),per_sample=results))

def spawn(label,gpu,smoke=False):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 d=O/label;d.mkdir(parents=True,exist_ok=True);log=(d/(f'smoke_gpu{gpu}.log' if smoke else f'worker{gpu}.log')).open('a')
 return subprocess.Popen([sys.executable,'-B',str(S/'run.py'),'worker','--model',label,'--gpu',str(gpu)]+(['--smoke'] if smoke else []),env=env,stdout=log,stderr=subprocess.STDOUT)

def manager(smoke=False):
 j=verify(full=True);locks=[];children=[]
 for path in [S/('smoke.lock' if smoke else 'manager.lock')]+[R/'work/downstream_gpu_ownership'/f'physical_gpu{g}.lock' for g in ([5] if smoke else GPUS)]:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 def stopped(sig,frame):raise KeyboardInterrupt('manager signal '+str(sig))
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 try:
  if smoke:
   free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=35000
   for label in j['models']:
    p=spawn(label,5,True);children=[p];assert p.wait()==0,('SMOKE_FAILED',label)
   atomic(O/'SMOKE_STATUS.json',dict(status='PASS',job_sha256=sha(S/'JOB.json')));return
  verify_import()
  for label in j['models']:
   s=read(O/label/'SMOKE.json');assert s['status']=='PASS' and s['job_sha256']==sha(S/'JOB.json')
  for gpu in GPUS:
   free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=35000,('GPU_HEADROOM',gpu,free)
  atomic(O/'PHASE_STATUS.json',dict(status='RUNNING',manager_pid=os.getpid(),gpus=GPUS,started=time.time(),job_sha256=sha(S/'JOB.json')))
  for label,c in j['models'].items():
   children=[spawn(label,gpu) for gpu in GPUS]
   atomic(O/label/'PHASE_STATUS.json',dict(status='RUNNING',pids=[p.pid for p in children],gpus=GPUS,started=time.time()))
   while any(p.poll() is None for p in children):
    assert all(p.poll() in [None,0] for p in children),('WORKER_FAILED',label,[(p.pid,p.poll()) for p in children])
    time.sleep(5)
   assert all(p.wait()==0 for p in children)
   aggregate(label,c);atomic(O/label/'PHASE_STATUS.json',dict(status='COMPLETE',ended=time.time()))
  atomic(O/'PHASE_STATUS.json',dict(status='COMPLETE',ended=time.time(),gpus=GPUS,job_sha256=sha(S/'JOB.json')))
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','import','manager','worker']);p.add_argument('--smoke',action='store_true');p.add_argument('--model');p.add_argument('--gpu',type=int);a=p.parse_args()
 try:
  if a.mode=='prepare':prepare()
  elif a.mode=='import':import_snapshot()
  elif a.mode=='manager':manager(a.smoke)
  else:worker(a.model,a.gpu,a.smoke)
 except BaseException as err:
  atomic(O/('ERROR_'+a.mode+'_'+str(a.model)+'_'+str(a.gpu)+'.json'),dict(error=repr(err),traceback=traceback.format_exc(),time=time.time()))
  if a.mode=='manager':atomic(O/('SMOKE_STATUS.json' if a.smoke else 'PHASE_STATUS.json'),dict(status='FAILED',error=repr(err),time=time.time()))
  raise
if __name__=='__main__':main()
