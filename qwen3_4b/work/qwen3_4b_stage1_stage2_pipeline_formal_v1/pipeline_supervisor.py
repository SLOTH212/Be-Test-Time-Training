#!/usr/bin/env python3
import argparse
from pipeline_core import *
def stage_snapshot(stage,c):return progress(stage,c)
def action_policy(stage,c,journal,snapshot):
 """Pure policy used by production and synthetic crash-window tests."""
 final=Path(c[f'stage{stage}']['final_root'])
 if final.exists():return 'VALIDATE_FINAL'
 if snapshot.get('process_identities'):return 'ADOPT_ACTIVE'
 if snapshot.get('completion'):return 'VALIDATE_COMPLETION'
 if journal is not None:return 'BLOCK_UNCERTAIN_OR_FAILED_PRIOR_ATTEMPT'
 if snapshot.get('run_root') and Path(snapshot['run_root']).exists():return 'BLOCK_EXISTING_RUN'
 if stage==2 and any((ROOT/'runs').glob('qwen3_4b_stage2_formal_*')):return 'BLOCK_ORPHAN_STAGE2_RUN'
 return 'LAUNCH_ONCE'
class Supervisor:
 def __init__(self,c,fd):
  self.c=c;self.root=owned(c['state_root']);self.fd=fd;self.path=self.root/'PIPELINE_STATE.json';self.state=read(self.path) if self.path.exists() else {'state':'PRECHECK','history':[]}
 def move(self,phase,stage,reason):
  if self.state['state']==phase:return
  spec=self.c[f'stage{stage}'];self.state=transition(self.state,phase,digest(spec['authority']),spec.get('run_root'),reason);atomic(self.path,self.state)
 def journal_path(self,stage):return self.root/f'STAGE{stage}_LAUNCH_RECEIPT.json'
 def launch(self,stage):
  path=self.journal_path(stage);require(not path.exists(),'DUPLICATE_LAUNCH_INTENT')
  spec=self.c[f'stage{stage}'];snap=stage_snapshot(stage,self.c);require(action_policy(stage,self.c,None,snap)=='LAUNCH_ONCE','RUN_ALREADY_EXISTS')
  if stage==2:
   parent=Path(self.c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';validate_final(parent.parent,1,self.c)
   command=['bash',spec['launcher'],'--parent-authority',str(parent)]
  else:command=[self.c['python'],'-B',spec['guard'],'launch']
  intent={'phase':'INTENT','stage':stage,'created_at':time.time(),'command':command,'authority_sha256':digest(spec['authority']),'prior_stage2_roots':sorted(str(p) for p in (ROOT/'runs').glob('qwen3_4b_stage2_formal_*')) if stage==2 else [],'source_authority_sha256':digest(SOURCE)}
  atomic(path,intent);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,self.c['physical_gpu_allowlist'])),WORLD_SIZE='2',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TMPDIR=str(ROOT/'tmp'))
  for key in ['CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']:env.pop(key,None)
  # Persist intent before spawn. The short launcher inherits the pipeline lock,
  # closing the supervisor-death / launcher-still-running race.
  log=self.root/f'STAGE{stage}_LAUNCH.log'
  with log.open('x') as f:
   proc=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(self.fd,))
   intent.update(launcher_identity=process_identity(proc.pid));atomic(path,intent);rc=proc.wait()
  intent.update(phase='LAUNCHER_RETURNED',returncode=rc,ended_at=time.time(),log_sha256=digest(log))
  snapshot=stage_snapshot(stage,self.c);intent['observed']=snapshot;atomic(path,intent)
  require(rc==0,'LAUNCHER_FAILED_NO_AUTOMATIC_RETRY');require(snapshot['run_root'],'MISSING_LAUNCH_RUN_RECEIPT')
 def step(self):
  from pipeline_finalize import finalize
  for stage in (1,2):
   spec=self.c[f'stage{stage}'];snapshot=stage_snapshot(stage,self.c);jp=self.journal_path(stage);journal=read(jp) if jp.exists() else None
   action=action_policy(stage,self.c,journal,snapshot)
   if action=='VALIDATE_FINAL':
    validate_final(spec['final_root'],stage,self.c)
    if self.state['state'] in ['BLOCKED','FAILED'] or STATES.index(self.state['state'])<STATES.index(f'STAGE{stage}_FINAL_FROZEN'):self.move(f'STAGE{stage}_FINAL_FROZEN',stage,'Verified existing immutable final; idempotent adoption')
    continue
   if action=='ADOPT_ACTIVE':
    if stage==2:
     parent=Path(self.c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';validate_final(parent.parent,1,self.c)
     require(read(Path(snapshot['run_root'])/'resolved_config.yaml')['input_identity']['parent_authority_sha256']==digest(parent),'ACTIVE_STAGE2_PARENT_AUTHORITY')
    if journal is None:atomic(jp,{'phase':'ADOPTED_EXISTING','observed':snapshot,'created_at':time.time(),'authority_sha256':digest(spec['authority'])})
    self.move(f'STAGE{stage}_RUNNING',stage,'Matching command and process start identity; no launch');return
   if action=='VALIDATE_COMPLETION':
    completion(snapshot['run_root'],stage,self.c)
    self.move(f'STAGE{stage}_FINALIZING',stage,'Process exited and exact counters/DCP/authorities verified')
    finalize(stage,snapshot['run_root'],self.c);self.move(f'STAGE{stage}_FINAL_FROZEN',stage,'Atomic immutable final verified');continue
   require(action=='LAUNCH_ONCE',action)
   if stage==2:validate_final(self.c['stage1']['final_root'],1,self.c)
   self.move(f'STAGE{stage}_READY',stage,'Prelaunch authority and parent gates passed');self.launch(stage);self.move(f'STAGE{stage}_RUNNING',stage,'Authorized launcher returned; durable launch receipt recorded');return
  self.finish()
 def finish(self):
  entries=[]
  for stage in (1,2):
   spec=self.c[f'stage{stage}'];a=validate_final(spec['final_root'],stage,self.c);path=Path(spec['final_root'])/f'STAGE{stage}_FINAL_AUTHORITY.json'
   entries.append({'class':f'STAGE{stage}_FINAL','model_path':spec['final_root'],'authority_path':str(path),'authority_sha256':digest(path),'model_manifest_sha256':digest(Path(spec['final_root'])/'SHA256SUMS.txt'),'parent_identity':a.get('stage1_final_authority_sha256',self.c['base_model_authority_sha256']),'training_data_identity':a['dataset_authority_sha256'],'config_identity':a['formal_config_sha256']})
  index={'FINAL_MODEL_ARTIFACT_N':2,'models':entries};ip=ROOT/'provenance/QWEN3_4B_TWO_STAGE_FINAL_MODEL_INDEX_V1.json'
  if ip.exists():require(read(ip)==index,'FINAL_INDEX_COLLISION')
  else:atomic(ip,index)
  self.move('PIPELINE_COMPLETE',2,'Exactly two immutable FINAL models verified; no downstream actions')
  authority={'status':'PIPELINE_COMPLETE','source_authority_sha256':digest(SOURCE),'state_history':self.state['history'],'state_sha256':digest(self.path),'final_model_index_path':str(ip),'final_model_index_sha256':digest(ip),'base_model_authority_sha256':self.c['base_model_authority_sha256'],'models':entries,'stage1_prelaunch_sha256':digest(self.c['stage1']['authority']),'stage2_preflight_sha256':digest(self.c['stage2']['authority']),'downstream_started':False}
  fp=ROOT/'provenance/QWEN3_4B_STAGE1_STAGE2_FORMAL_PIPELINE_FINAL_AUTHORITY_V1.json'
  if fp.exists():require(read(fp)==authority,'PIPELINE_FINAL_AUTHORITY_COLLISION')
  else:atomic(fp,authority);fp.with_suffix('.sha256').write_text(digest(fp)+'  '+fp.name+'\n')
 def run(self):
  atomic(self.root/'SUPERVISOR_PROCESS.json',{'identity':process_identity(os.getpid()),'source_authority_sha256':digest(SOURCE),'started_at':time.time()})
  try:
   while True:
    self.step();atomic(self.root/'PIPELINE_HEARTBEAT.json',{'timestamp':time.time(),'phase':self.state['state'],'stage1':stage_snapshot(1,self.c),'stage2':stage_snapshot(2,self.c),'supervisor':process_identity(os.getpid())})
    if self.state['state']=='PIPELINE_COMPLETE':return
    time.sleep(self.c['heartbeat_seconds'])
  except Exception as e:
   self.state=transition(self.state,'BLOCKED',digest(SOURCE),self.root,str(e));self.state['latest_error']={'type':type(e).__name__,'message':str(e),'time':time.time()};atomic(self.path,self.state);raise
def status(c):
 root=Path(c['state_root']);result={'PIPELINE_STATE':read(root/'PIPELINE_STATE.json') if (root/'PIPELINE_STATE.json').exists() else {'state':'NOT_STARTED'},'stage1':progress(1,c),'stage2':progress(2,c)}
 for stage in (1,2):
  path=Path(c[f'stage{stage}']['final_root']);item={'exists':path.exists(),'root':str(path),'authority':str(path/f'STAGE{stage}_FINAL_AUTHORITY.json'),'integrity':'NOT_CREATED'}
  if path.exists():
   try:validate_final(path,stage,c);item['integrity']='PASS'
   except Exception as e:item['integrity']='FAIL';item['error']=str(e)
  result[f'stage{stage}_final']=item
 try:result['GPU4_GPU5']=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,c['physical_gpu_allowlist'])),'--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv'],text=True,timeout=15)
 except Exception as e:result['gpu_query_error']=str(e)
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--status',action='store_true');ap.add_argument('--recover',action='store_true');ap.add_argument('--supervise',action='store_true');args=ap.parse_args();c=config()
 require(sum(map(int,[args.execute,args.status,args.recover,args.supervise]))<=1,'CONFLICTING_ACTIONS')
 if args.status:print(json.dumps(status(c),indent=2));return
 source_contract()
 if not any([args.execute,args.recover,args.supervise]):
  check=preflight(c,verify_guard=True);print(json.dumps({'mode':'DRY_RUN','preflight':check,'planned_states':STATES,'Stage1_command':[c['python'],c['stage1']['guard'],'launch'],'Stage2_command':[c['stage2']['launcher'],'--parent-authority',str(Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json')],'final_model_roots':[c[f'stage{s}']['final_root'] for s in (1,2)],'formal_training_started':False},indent=2));return
 root=owned(c['state_root']);root.mkdir(parents=True,exist_ok=True)
 with lock(root/'PIPELINE.lock') as fd:
  token=root/'EXECUTION_AUTHORIZED.json'
  if args.execute:
   require(not token.exists(),'PIPELINE_ALREADY_AUTHORIZED_USE_RECOVERY');preflight(c,verify_guard=True)
   atomic(token,{'explicit_execute':True,'created_at':time.time(),'source_authority_sha256':digest(SOURCE),'config_sha256':digest(CONFIG)})
  else:require(token.exists() and read(token).get('explicit_execute') is True,'NO_PRIOR_EXPLICIT_EXECUTE_AUTHORIZATION')
  require(read(token)['source_authority_sha256']==digest(SOURCE) and read(token)['config_sha256']==digest(CONFIG),'EXECUTION_AUTHORITY_CHANGED')
  if args.supervise:preflight(c);Supervisor(c,fd).run();return
  # Parent retains lock until child exec. Child inherits the SAME flock then
  # closes it immediately before main re-acquires; existing intent prevents duplicates.
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',WORLD_SIZE='2')
  log=(root/'supervisor.log').open('a')
  cmd=[c['python'],'-B',c['supervisor'],'--supervise']
  # Lock is released on this parent exit. Child waits briefly only for startup,
  # retries lock contention without ever invoking a trainer outside the lock.
  bootstrap="import os,time; time.sleep(1); os.execv("+repr(c['python'])+","+repr(cmd)+")"
  proc=subprocess.Popen([c['python'],'-c',bootstrap],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=ROOT)
  print(json.dumps({'detached_supervisor_pid':proc.pid,'state_root':str(root),'mode':'SUPERVISOR_RECOVERY' if args.recover else 'EXPLICIT_EXECUTE'}))
if __name__=='__main__':main()
