"""Detached exactly-once orchestration, with no training mathematics."""
import argparse,signal,sys
from core import *
def env(gpus='0,1,2,3,4'):return {**os.environ,'CUDA_VISIBLE_DEVICES':gpus,'WORLD_SIZE':'5','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
def policy(run_exists,journal,active,complete,final):
 if final:return 'FINAL'
 if active:return 'ADOPT'
 if complete:return 'COMPLETE'
 if journal is not None or run_exists:return 'BLOCK'
 return 'LAUNCH'
def move(c,state,reason):
 root=Path(c['state_root']);p=root/'PIPELINE_STATE.json';old=read(p) if p.exists() else {'state':'PRECHECK','history':[]}
 if old['state']==state:return
 require(state in STATES+['BLOCKED','FAILED'],'STATE_NAME')
 if state not in ['BLOCKED','FAILED'] and old['state'] not in ['BLOCKED','FAILED']:require(STATES.index(state)>=STATES.index(old['state']),'STATE_REGRESSION')
 atomic(p,{'state':state,'history':old['history']+[{'previous':old['state'],'state':state,'time':time.time(),'reason':reason,'source_sha256':sha(SOURCE)}],'updated':time.time()})
def stop_own(proc,ident):
 if proc.poll() is None and alive(ident):
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=45)
  except subprocess.TimeoutExpired:
   if alive(ident):os.killpg(proc.pid,signal.SIGKILL)
   proc.wait()
def trainer(stage,c,parent,resume=False):
 run=Path(c[f'stage{stage}']['run_root']);root=Path(c['state_root']);journal=root/f'STAGE{stage}_LAUNCH_RECEIPT.json'
 require(stage==2,'STAGE1_ALREADY_FROZEN');require(journal.exists(),'NO_LAUNCH_INTENT');require(run.exists() if resume else not run.exists(),'TRAINER_ROOT_CONTRACT')
 with lock(root/f'STAGE{stage}.lock'):
  if not resume:run.mkdir()
  receipt={'manager':identity(os.getpid()),'started':time.time(),'stage':stage,'source_sha256':sha(SOURCE)};atomic(run/'process.json',receipt)
  proc=None;ident=None;rc=-1;error=None
  try:
   preflight(c,live=True)
   command=[str(R/'envs/ttt_runtime_v1/bin/torchrun'),'--standalone','--nproc-per-node=5','--max-restarts=0',str(W/'entry.py'),'--stage',str(stage),'--config',c[f'stage{stage}']['config'],'--output-dir',str(run)]
   if resume:command+=['--resume-from',str(run/'checkpoints')]
   if stage==2:
    required=Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';require(parent and Path(parent)==required,'STAGE2_PARENT_AUTHORITY_REQUIRED');validate_final(required.parent,1,c);command+=['--parent-artifact',str(required.parent)];receipt['parent_authority_sha256']=sha(required)
   with (run/'launcher.log').open('x') as log:
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**env(),**({'LLAMA_RESUME_VERIFY_SAVE':'1'} if resume else {})},cwd=R,start_new_session=True)
    ident=identity(proc.pid);receipt.update(torchrun=ident,command=command);atomic(run/'process.json',receipt);last_step=0
    while proc.poll() is None:
     workers=[]
     for p in Path('/proc').iterdir():
      if p.name.isdigit():
       x=identity(p.name)
       if x and str(W/'entry.py') in x['command'] and str(run) in x['command'] and x['pid']!=proc.pid:workers.append(x)
     if workers:receipt['workers']=workers;atomic(run/'process.json',receipt)
     row=last(run)
     if row and row['update_step']>last_step:
      health(row,stage,c);last_step=row['update_step']
      if stage==1 and last_step>=3 and not (run/'startup_health.json').exists():atomic(run/'startup_health.json',{'status':'PASS','same_formal_run':True,'step':last_step,'row':row,'process':receipt,'validation_record_seen_in_training_n':0})
     time.sleep(3)
    rc=proc.wait()
   require(rc==0,'TRAINER_NONZERO_EXIT')
   row=last(run);require(row is not None,'NO_TRAINING_RECEIPTS');health(row,stage,c)
  except BaseException as e:
   error=repr(e)
   if proc is not None:stop_own(proc,ident);rc=proc.returncode
   move(c,'BLOCKED',f'Stage{stage} trainer failure: '+error)
  finally:atomic(run/'launcher_exit.json',{'exit_code':rc,'error':error,'ended':time.time(),'manager':identity(os.getpid()),'no_automatic_retry':True})
  if error:raise RuntimeError(error)
def launch_stage(stage,c,parent=None):
 root=Path(c['state_root']);p=root/f'STAGE{stage}_LAUNCH_RECEIPT.json';run=Path(c[f'stage{stage}']['run_root'])
 require(not p.exists() and not run.exists(),'DUPLICATE_STAGE_LAUNCH')
 if stage==2:validate_final(c['stage1']['final_root'],1,c)
 atomic(p,{'phase':'INTENT','stage':stage,'time':time.time(),'source_sha256':sha(SOURCE),'run_root':str(run)})
 command=[sys.executable,'-B',str(W/'pipeline.py'),'trainer','--stage',str(stage),'--execute']
 if stage==2:command+=['--parent-authority',str(Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json')]
 with (root/f'stage{stage}_manager.log').open('a') as log:proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env(''),cwd=R)
 rec=read(p);rec.update(phase='SPAWNED',manager=identity(proc.pid));atomic(p,rec)
def supervise(c):
 root=Path(c['state_root'])
 with lock(root/'PIPELINE.lock'):
  require((root/'EXECUTION_AUTHORIZED.json').exists(),'NO_EXECUTE_AUTHORITY');require(read(root/'EXECUTION_AUTHORIZED.json')['source_sha256']==sha(SOURCE),'AUTHORITY_CHANGED');source();atomic(root/'SUPERVISOR_PROCESS.json',identity(os.getpid()))
  try:
   while True:
    for stage in (1,2):
     spec=c[f'stage{stage}'];snap=trainer_snapshot(stage,c);jp=root/f'STAGE{stage}_LAUNCH_RECEIPT.json';j=read(jp) if jp.exists() else None
     active=bool(snap['active']) or bool(j and alive(j.get('manager')))
     action=policy(Path(spec['run_root']).exists(),j,active,snap['complete'],Path(spec['final_root']).exists())
     if action=='FINAL':validate_final(spec['final_root'],stage,c);continue
     if action=='ADOPT':move(c,f'STAGE{stage}_RUNNING','Existing identified trainer adopted without relaunch');break
     if action=='COMPLETE':
      completion(stage,c);move(c,f'STAGE{stage}_FINALIZING','Exact counters and final DCP verified')
      from finalizer import finalize
      finalize(stage,c);move(c,f'STAGE{stage}_FINAL_FROZEN','Atomic standalone FINAL verified');continue
     require(action=='LAUNCH','FAILED_OR_UNCERTAIN_ATTEMPT_NO_AUTO_RETRY')
     if stage==2:validate_final(c['stage1']['final_root'],1,c)
     move(c,f'STAGE{stage}_READY','Prelaunch gates passed');launch_stage(stage,c);move(c,f'STAGE{stage}_RUNNING','Exactly-once trainer manager dispatched');break
    else:
     models=[]
     for s in (1,2):
      validate_final(c[f'stage{s}']['final_root'],s,c);p=Path(c[f'stage{s}']['final_root'])/f'STAGE{s}_FINAL_AUTHORITY.json';models.append({'stage':s,'root':str(p.parent),'authority':str(p),'sha256':sha(p)})
     atomic(R/'provenance/LLAMA31_8B_STAGE2_FIVE_GPU_FINAL_MODEL_INDEX_V3.json',{'FINAL_MODEL_ARTIFACT_N':2,'models':models});move(c,'PIPELINE_COMPLETE','Two FINAL models verified; stop without downstream execution')
     atomic(R/'provenance/LLAMA31_8B_STAGE2_FIVE_GPU_PIPELINE_FINAL_AUTHORITY_V3.json',{'status':'PIPELINE_COMPLETE','models':models,'source_sha256':sha(SOURCE),'bindings':c['bindings'],'state':read(root/'PIPELINE_STATE.json'),'DOWNSTREAM_EXPERIMENTS_AUTO_START':False});return
    atomic(root/'PIPELINE_HEARTBEAT.json',{'time':time.time(),'state':read(root/'PIPELINE_STATE.json'),'supervisor':identity(os.getpid()),'stage1':trainer_snapshot(1,c),'stage2':trainer_snapshot(2,c)})
    time.sleep(300)
  except BaseException as e:move(c,'BLOCKED',repr(e));raise
def status(c):
 root=Path(c['state_root']);result={'PIPELINE_STATE':read(root/'PIPELINE_STATE.json') if (root/'PIPELINE_STATE.json').exists() else {'state':'NOT_STARTED'},'stage1_dataset_metadata':SPLIT}
 for stage in [1,2]:
  snap=trainer_snapshot(stage,c);row=snap.pop('row');item={**snap,'expected':EXPECTED[stage],'records':row['record_cursor'] if row else 0,'tokens':row['cumulative_tokens'] if row else 0,'optimizer_steps':row['update_step'] if row else 0,'loss':row['global_mean_loss'] if row else None,'latest_DCP':None}
  if row:
   seconds=row['iteration_seconds_excluding_checkpoint'];rate=sum(r['local_tokens'] for r in row['rank_runtime'])/seconds;item.update(throughput=rate,ETA_seconds=(EXPECTED[stage]['cumulative_tokens']-row['cumulative_tokens'])/rate,step_seconds=seconds)
  p=Path(snap['run_root'])/'checkpoints/latest.json'
  if p.exists():item['latest_DCP']=read(p)
  final=Path(c[f'stage{stage}']['final_root']);item['final']={'exists':final.exists(),'root':str(final),'integrity':'NOT_CREATED'}
  if final.exists():
   try:validate_final(final,stage,c);item['final']['integrity']='PASS'
   except Exception as e:item['final'].update(integrity='FAIL',error=str(e))
  result[f'stage{stage}']=item
 result['GPU0_4']=subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
 result['QWEN4B_PIPELINE_STATE']=json.loads(subprocess.check_output([str(R/'bin/status_qwen3_4b_stage1_stage2_pipeline_formal_v1.sh')],text=True))['PIPELINE_STATE']['state'];return result
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','recover','supervise','trainer','status','verify','finalize']);p.add_argument('--execute',action='store_true');p.add_argument('--stage',type=int,choices=[1,2]);p.add_argument('--parent-authority');p.add_argument('--resume',action='store_true');a=p.parse_args();c=config();root=Path(c['state_root'])
 if a.mode=='status':print(json.dumps(status(c),indent=2));return
 if a.mode=='verify':print(json.dumps(preflight(c,live=True)));return
 if a.mode=='finalize':
  from finalizer import finalize
  finalize(a.stage,c);return
 if a.mode=='trainer':require(a.execute and a.stage,'EXPLICIT_EXECUTE_REQUIRED');trainer(a.stage,c,a.parent_authority,a.resume);return
 if a.mode=='resume':
  require(a.execute,'EXPLICIT_RESUME_EXECUTE_REQUIRED')
  from resume_v2 import dispatch
  dispatch(c);return
 if a.mode=='supervise':supervise(c);return
 if a.mode=='prune':
  require(read(root/'PIPELINE_STATE.json')['state']=='PIPELINE_COMPLETE','PIPELINE_NOT_COMPLETE')
  for s in [1,2]:validate_final(c[f'stage{s}']['final_root'],s,c)
  targets=[Path(c[f'stage{s}']['run_root'])/'checkpoints' for s in [1,2]];print(json.dumps({'dry_run':not a.execute,'targets':[str(t) for t in targets]}))
  if a.execute:
   for t in targets:require(t.resolve().parent==Path(c['stage1']['run_root']) or t.resolve().parent==Path(c['stage2']['run_root']),'PRUNE_PATH');shutil.rmtree(t)
  return
 if a.mode=='launch' and not a.execute:print(json.dumps({'mode':'DRY_RUN','preflight':preflight(c,live=True),'states':STATES}));return
 with lock(root/'PIPELINE.lock'):
  token=root/'EXECUTION_AUTHORIZED.json'
  if a.mode=='launch':
   require(a.execute and not token.exists(),'DUPLICATE_PIPELINE_LAUNCH');preflight(c,live=True)
   validate_final(c['stage1']['final_root'],1,c);require(not Path(c['stage2']['run_root']).exists(),'FORMAL_RUN_COLLISION');atomic(token,{'source_sha256':sha(SOURCE),'time':time.time(),'execute':True})
  else:require(token.exists(),'NO_PRIOR_EXECUTION_AUTHORIZATION')
  with (root/'supervisor.log').open('a') as log:
   cmd=[sys.executable,'-B',str(W/'pipeline.py'),'supervise'];bootstrap='import os,time;time.sleep(1);os.execv('+repr(sys.executable)+','+repr(cmd)+')';proc=subprocess.Popen([sys.executable,'-c',bootstrap],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env(''),cwd=R,start_new_session=True)
  print(json.dumps({'detached_supervisor_pid':proc.pid,'state_root':str(root)}))
if __name__=='__main__':main()
