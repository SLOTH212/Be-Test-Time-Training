import os,sys,time,pathlib,subprocess,json
R=pathlib.Path('/path/to/ttt');sys.path.insert(0,str(R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1'))
import pipeline_core as core
import pipeline_finalize as fin
import pipeline_supervisor as sup

def freeze_after_rename(temp,final):
 core.require(not final.exists(),'REFUSE_FINAL_OVERWRITE')
 for p in temp.rglob('*'):
  core.require(not p.is_symlink(),'FINAL_SYMLINK_FORBIDDEN')
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
   p.chmod(0o444)
 for p in sorted([p for p in temp.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
 temp.chmod(0o755);core.fsync_dir(temp)
 try:os.rename(temp,final)
 finally:
  if final.exists():final.chmod(0o555)
  elif temp.exists():temp.chmod(0o555)
 core.fsync_dir(final);core.fsync_dir(final.parent)
fin.freeze_tree=freeze_after_rename
original_step=sup.Supervisor.step

def capacity_wait_step(self):
 s1=pathlib.Path(self.c['stage1']['final_root']);s2=pathlib.Path(self.c['stage2']['final_root']);jp=self.journal_path(2)
 if s1.exists() and not s2.exists() and not jp.exists():
  snapshot=sup.stage_snapshot(2,self.c)
  if sup.action_policy(2,self.c,None,snapshot)=='LAUNCH_ONCE':
   out=subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
   values={int(a):int(b) for a,b in [line.split(',') for line in out.strip().splitlines()]}
   core.require(set(values)=={4,5},'GPU_SET_MISMATCH')
   wait=any(v<48000 for v in values.values());core.atomic(self.root/'STAGE2_RESOURCE_WAIT.json',{'status':'WAITING_FOR_CAPACITY' if wait else 'CAPACITY_READY','timestamp':time.time(),'free_mib':values,'required_free_mib_per_gpu':48000,'automatic_start_when_available':True,'foreign_processes_modified':False})
   if wait:
    if self.state['state']!='STAGE2_READY':
     core.validate_final(s1,1,self.c);self.move('STAGE1_FINAL_FROZEN',1,'Published Stage1 final validated after permission repair');self.move('STAGE2_READY',2,'Waiting for existing GPU4/5 capacity threshold before launch intent')
     self.state.pop('latest_error',None);core.atomic(self.path,self.state)
    return
 return original_step(self)
sup.Supervisor.step=capacity_wait_step
if __name__=='__main__':
 cfg=core.config();core.source_contract();core.atomic(pathlib.Path(cfg['state_root'])/'REPAIRED_SUPERVISOR_RUNTIME.json',{'script':__file__,'sha256':core.digest(__file__),'frozen_source_authority_sha256':core.digest(core.SOURCE),'changes':['Move staging root while owner-writable, then freeze published root','Wait for unchanged Stage2 48000 MiB capacity gate before recording launch intent'],'scientific_configuration_changed':False,'created_at':time.time()});sup.main()
