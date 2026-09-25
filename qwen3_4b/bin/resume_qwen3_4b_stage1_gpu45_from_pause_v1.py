import os,sys,json,time,pathlib,subprocess,fcntl,hashlib,shutil,traceback
R=pathlib.Path('/path/to/ttt'); W=R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1'; sys.path.insert(0,str(W))
import pipeline_core as core
c=core.config(); run=pathlib.Path(c['stage1']['run_root']); state=pathlib.Path(c['state_root']); stamp=time.strftime('%Y%m%dT%H%M%S'); receipt=state/('RESUME_'+stamp+'.json')
def atomic(p,o):core.atomic(p,o)
info={'status':'PREFLIGHT','created_at':time.time(),'explicit_user_resume':True,'resume_from':str(run/'checkpoints'),'physical_gpus':[4,5],'scientific_config_changed':False}
atomic(receipt,info)
try:
 with core.lock(state/'PIPELINE.lock'):
  fd=os.open(R/'runs'/(run.name+'.lock'),os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not core.matching_processes(run),'EXISTING_TRAINER'
  assert core.read(state/'USER_PAUSE_RECEIPT.json')['status']=='PAUSED'
  core.source_contract();core.preflight(c)
  guard=core.module(c['stage1']['guard'],'resume_guard');cfg,auth=guard.guard()
  cp=core.dcp_contract(run,full_hash=True); latest=core.read(run/'checkpoints/latest.json'); assert latest['progress']['update_step']==7000
  pause=core.read(state/'USER_PAUSE_RECEIPT.json');assert latest==pause['checkpoint'],'CHECKPOINT_CHANGED'
  archive=run/('attempt_before_resume_'+stamp);archive.mkdir()
  for name in ['training.jsonl','launcher.log','launcher_exit.json','supervisor.log']:
   src=run/name
   if src.exists():shutil.copy2(src,archive/name)
  pointer=R/'runs'/(run.name+'.process.json');shutil.copy2(pointer,archive/'process.json');shutil.copy2(state/'USER_PAUSE_RECEIPT.json',archive/'USER_PAUSE_RECEIPT.json')
  hashes={str(p.relative_to(archive)):core.digest(p) for p in archive.iterdir() if p.is_file()};atomic(archive/'MANIFEST.json',{'files':hashes,'resume_boundary':latest,'reason':'User-authorized checkpoint resume; original attempt preserved intact'})
  retained=[];discarded=[]
  for line in (archive/'training.jsonl').read_text().splitlines():
   x=json.loads(line)
   (retained if x['update_step']<=7000 else discarded).append(line)
  assert len(retained)==7000 and json.loads(retained[-1])['cumulative_tokens']==latest['progress']['cumulative_tokens']
  temp=run/'training.resume_tmp';temp.write_text('\n'.join(retained)+'\n');temp.replace(run/'training.jsonl')
  (run/'launcher_exit.json').unlink()
  command=[str(R/'envs/ttt_runtime_v1/bin/torchrun'),'--standalone','--nproc-per-node=2',str(R/'bin/stage1_runtime_entry_gpu45_pipeline_v1.py'),'--stage','1','--config',str(c['stage1']['config']),'--output-dir',str(run),'--resume-from',str(run/'checkpoints')]
  info.update(status='LAUNCH_INTENT',archive=str(archive),checkpoint=latest,replay_logged_steps=len(discarded),command=command,script_sha256=core.digest(__file__));atomic(receipt,info)
  log=(run/'launcher.log').open('w');started=time.time();proc=subprocess.Popen(command,cwd=R,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  atomic(pointer,{'supervisor_pid':os.getpid(),'torchrun_pid':proc.pid,'torchrun_identity':guard.process_identity(proc.pid),'started':started,'command':command,'FORMAL_STAGE1_TRAINING_STARTED':True,'resumed_from_step':7000,'resume_receipt':str(receipt),'previous_attempt_elapsed_seconds':core.read(archive/'launcher_exit.json')['elapsed_seconds']})
  info.update(status='RUNNING',started_at=started,torchrun=core.process_identity(proc.pid));atomic(receipt,info)
  pause.update(status='RESUMED',resumed_at=started,resume_receipt=str(receipt));atomic(state/'USER_PAUSE_RECEIPT.json',pause)
  s=core.read(state/'PIPELINE_STATE.json');s.pop('pause_status',None);s.pop('latest_error',None);s=core.transition(s,'STAGE1_RUNNING',auth,run,'Explicit user resume from hash-verified step 7000; new attempt archived separately');atomic(state/'PIPELINE_STATE.json',s)
 print('RESUME_RUNNING '+str(receipt),flush=True)
 rc=proc.wait();log.close();complete=run/'complete.json';valid=rc==0 and complete.exists()
 if valid:
  done=core.read(complete);valid=done.get('complete') is True and done.get('record_cursor')==cfg['record_cursor_final'] and done.get('cumulative_tokens')==cfg['train_tokens']
 atomic(run/'launcher_exit.json',{'exit_code':rc,'complete_verified':valid,'ended':time.time(),'elapsed_seconds':time.time()-started,'resume_from_step':7000,'resume_receipt':str(receipt)})
 info.update(status='COMPLETE' if valid else 'FAILED',exit_code=rc,ended_at=time.time());atomic(receipt,info);os.close(fd)
except Exception as e:
 info.update(status='FAILED',error=repr(e),ended_at=time.time());atomic(receipt,info);traceback.print_exc();raise
