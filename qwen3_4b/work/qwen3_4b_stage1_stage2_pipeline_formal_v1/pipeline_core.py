"""Authority-driven orchestration utilities. No training mathematics or CUDA imports."""
import contextlib,fcntl,hashlib,importlib.util,json,os,pwd,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/path/to/ttt')
CONFIG=ROOT/'configs/qwen3_4b_stage1_stage2_pipeline_formal_v1.yaml'
SOURCE=ROOT/'provenance/QWEN3_4B_STAGE1_STAGE2_PIPELINE_SOURCE_AUTHORITY_V1.json'
STATES=['PRECHECK','STAGE1_READY','STAGE1_RUNNING','STAGE1_FINALIZING','STAGE1_FINAL_FROZEN','STAGE2_READY','STAGE2_RUNNING','STAGE2_FINALIZING','STAGE2_FINAL_FROZEN','PIPELINE_COMPLETE']
class ContractError(RuntimeError):pass
def require(test,message):
 if not test:raise ContractError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def owned(path):
 p=Path(path);require(not p.is_symlink(),'SYMLINK_ROOT_FORBIDDEN '+str(p));p=p.resolve()
 require(p.is_relative_to(ROOT) and p!=ROOT,'PATH_OUTSIDE_TASK_ROOT '+str(p));return p
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic(path,data):
 path=owned(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fsync_dir(path.parent)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def config():return read(CONFIG)
def process_identity(pid):
 try:
  p=Path('/proc')/str(int(pid));stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0]=='Z':return None
  return {'pid':int(pid),'start_ticks':int(stat[19]),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 except (OSError,ValueError,TypeError):return None
def alive(identity):return bool(identity and process_identity(identity.get('pid'))==identity)
def matching_processes(run_root):
 token=str(run_root);result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  row=process_identity(p.name)
  if row and token in row['command'] and any(x in row['command'] for x in ['torchrun','distributed_train_worker','stage1_runtime_entry','stage2_runtime_entry']):result.append(row)
 return result
@contextlib.contextmanager
def lock(path):
 path=owned(path);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ContractError('SUPERVISOR_ALREADY_ACTIVE')
  yield fd
 finally:os.close(fd)
def check_hashes(files):
 for path,h in files.items():require(digest(path)==h,'BOUND_FILE_HASH_MISMATCH '+path)
def source_contract():
 a=read(SOURCE);require(digest(SOURCE)==SOURCE.with_suffix('.sha256').read_text().split()[0],'PIPELINE_SOURCE_AUTHORITY_HASH');check_hashes(a['files']);return a
def preflight(c,verify_guard=False):
 require(os.uname().nodename=='amax' and pwd.getpwuid(os.getuid()).pw_name=='USER','HOST_ACCOUNT_IDENTITY')
 check_hashes(c['bindings']);a1=read(c['stage1']['authority']);a2=read(c['stage2']['authority'])
 s1=module(c['stage1']['guard'],'pipeline_s1_guard');s1.validate_v4_1_authority(a1)
 require(a2['QWEN3_4B_STAGE2_PREFLIGHT_STATUS']=='PASS_WITH_WARNING' and a2['SAFE_TO_RUN_FORMAL_STAGE2_AFTER_VALID_STAGE1_PARENT']=='YES','STAGE2_PREFLIGHT_NOT_CLEARED')
 check_hashes(a1['file_sha256']);check_hashes(a2['file_sha256'])
 s2=module(c['stage2']['controller'],'pipeline_s2_guard');s2.data_contract()
 import yaml
 f1=yaml.safe_load(Path(c['stage1']['config']).read_text());f2=read(c['stage2']['config'])
 for stage,cfg in [(1,f1),(2,f2)]:
  e=c['stage'+str(stage)]['expected'];require(cfg['record_cursor_final']==e['record_cursor'] and cfg['train_tokens']==e['cumulative_tokens'],'CONFIG_COUNTER_MISMATCH')
  batch=cfg['micro_batch_size']*cfg['world_size']*cfg['gradient_accumulation'];require(batch==2 and (cfg['record_cursor_final']+batch-1)//batch==e['update_step'],'EXPECTED_OPTIMIZER_STEPS')
  require(cfg['ttt_chunk_size']==4096 and cfg['context_length']==32768 and cfg['ttt_layers']==[0,6,12,18,24,30],'SCIENTIFIC_CONTRACT')
 require(f1['validation_tokens']==9999975,'VALIDATION_TOKEN_CONTRACT')
 require(a2['STAGE2_ANSWER_POSITIONS']==388806 and a2['STAGE2_CONTEXT_POSITIONS']==29610269,'STAGE2_POSITION_CONTRACT')
 require(c['stage1']['final_root']!=c['stage2']['final_root'],'FINAL_PATH_COLLISION')
 for stage in (1,2):
  final=owned(c[f'stage{stage}']['final_root'])
  if final.exists():validate_final(final,stage,c)
 storage(c)
 if verify_guard:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,c['physical_gpu_allowlist'])),WORLD_SIZE='2',PYTHONDONTWRITEBYTECODE='1')
  subprocess.run([c['python'],'-B',c['stage1']['guard'],'verify'],env=env,check=True)
 return {'status':'PASS','stage1':'V4.1','stage2':'PASS_WITH_WARNING','formal_training_started':False,'CUDA_initialized':False}
def storage(c):
 required=c['storage_required_bytes'];free=shutil.disk_usage(ROOT).free;require(free>=required,'INSUFFICIENT_CHECKPOINT_AND_FINAL_STORAGE');return {'free_bytes':free,'required_bytes':required}
def last_row(path):
 if not Path(path).exists():return None
 with Path(path).open('rb') as f:
  f.seek(max(0,Path(path).stat().st_size-(8<<20)));lines=f.read().splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
 return None
def dcp_contract(run,expected=None,full_hash=True):
 run=owned(run);root=run/'checkpoints';latest=read(root/'latest.json');slot=owned(root/latest['slot']);require(slot.parent==root,'DCP_SLOT_ESCAPE')
 require(digest(slot/'manifest.json')==latest['manifest_sha256'],'DCP_MANIFEST_HASH');m=read(slot/'manifest.json');require(m.get('complete'),'INCOMPLETE_DCP')
 if expected:
  for key,value in expected.items():require(m['progress'].get(key)==value,'FINAL_DCP_COUNTER_'+key)
 for name,h in m['files'].items():
  path=(slot/name).resolve();require(path.is_relative_to(slot) and not (slot/name).is_symlink(),'DCP_FILE_ESCAPE')
  if full_hash:require(digest(path)==h,'DCP_FILE_HASH '+name)
 return {'checkpoint_path':str(slot),'checkpoint_manifest_sha256':digest(slot/'manifest.json'),'progress':m['progress'],'manifest':m,'latest_sha256':digest(root/'latest.json')}
def completion(run,stage,c,full_hash=True):
 run=owned(run);require(not matching_processes(run),'TRAINING_PROCESS_STILL_ACTIVE');done=read(run/'complete.json');expected=c[f'stage{stage}']['expected']
 require(done.get('complete') is True and done.get('stage')==stage,'FINAL_COMPLETION_MARKER_REQUIRED')
 for k in ['record_cursor','cumulative_tokens']:require(done.get(k)==expected[k],'COMPLETION_'+k)
 last=last_row(run/'training.jsonl');require(last is not None,'FINAL_TRAINING_RECEIPT_MISSING')
 for k,v in expected.items():require(last.get(k)==v,'FINAL_TRAINING_'+k)
 require(all(x.get('weights_finite') is True for x in last['rank_runtime']),'FINAL_WEIGHTS_NOT_FINITE')
 if stage==1:
  exit=read(run/'launcher_exit.json');require(exit.get('exit_code')==0 and exit.get('complete_verified') is True,'STAGE1_EXIT_NOT_VERIFIED')
 log=(run/'launcher.log').read_text(errors='replace')
 require('FINAL_BARRIER_PASS' in log and all(f'RANK_{r}_SHUTDOWN_PASS' in log for r in (0,1)),'FINAL_CLEAN_SHUTDOWN_REQUIRED')
 require(not any(x in log for x in ['Traceback (most recent call last)','RuntimeError:','CUDA out of memory','OutOfMemoryError:']),'TRAINER_ERROR_LOG')
 cp=dcp_contract(run,expected,full_hash);require(cp['progress']['stage']==stage,'DCP_STAGE')
 import yaml
 cfgpath=Path(c[f'stage{stage}']['config']) if stage==1 else run/'resolved_config.yaml';cfg=yaml.safe_load(cfgpath.read_text())
 require(cp['progress']['input_identity']=={**cfg['input_identity'],'config_sha256':digest(cfgpath)} or all(cp['progress']['input_identity'].get(k)==v for k,v in cfg['input_identity'].items()),'DCP_INPUT_AUTHORITY_MISMATCH')
 require(cp['progress']['input_identity'].get('config_sha256')==digest(cfgpath),'RUN_CONFIG_IDENTITY')
 return {'stage':stage,'stage_completed':True,'run_root':str(run),'completion_sha256':digest(run/'complete.json'),'training_log_sha256':digest(run/'training.jsonl'),'launcher_log_sha256':digest(run/'launcher.log'),'config_sha256':digest(cfgpath),**expected,**cp}
def validate_final(root,stage,c):
 root=owned(root);path=root/f'STAGE{stage}_FINAL_AUTHORITY.json';a=read(path)
 require(a.get('artifact_class')==f'STAGE{stage}_FINAL' and a.get('immutable') is True and not a.get('DEBUG_ONLY'),'FINAL_CLASS')
 require(a.get(f'STAGE{stage}_FINAL_INTEGRITY')=='PASS','FINAL_INTEGRITY')
 require(a.get('final_root')==str(root),'FINAL_ROOT_IDENTITY')
 require(a.get('prelaunch_authority_sha256')==digest(c[f'stage{stage}']['authority']),'FINAL_PRELAUNCH_IDENTITY')
 for k,v in c[f'stage{stage}']['expected'].items():require(a.get(k)==v,'FINAL_AUTHORITY_COUNTER_'+k)
 files=a['files'];require('SHA256SUMS.txt' in files and 'config.json' in files and 'tokenizer.json' in files,'FINAL_STANDALONE_FILES')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()};require(actual==set(files)|{path.name},'UNBOUND_FINAL_FILES')
 for rel,h in files.items():
  p=root/rel;require(not p.is_symlink() and p.resolve().is_relative_to(root),'FINAL_SYMLINK_OR_ESCAPE');require(digest(p)==h,'FINAL_HASH_MISMATCH '+rel)
 require(not any('optimizer' in x or x.endswith('.distcp') or x.startswith('checkpoints/') for x in actual),'RECOVERY_PAYLOAD_IN_FINAL')
 if stage==1:module(c['stage2']['controller'],'final_parent_parser').parent_contract(path)
 else:
  parent=Path(a['stage1_final_authority']);require(digest(parent)==a['stage1_final_authority_sha256'],'STAGE1_PARENT_MUTATED')
  cfg=read(root/'config.json');require(cfg['ttt_chunk']==4096 and cfg['ttt_lr']==1.0 and cfg['ttt_update_clip_norm']==1e-5,'INFERENCE_4K_CONTRACT')
 return a
def progress(stage,c):
 spec=c[f'stage{stage}'];run=Path(spec['run_root']) if stage==1 else None
 if stage==2 and Path(spec['pointer']).exists():run=owned(read(spec['pointer'])['run_root'])
 result={'run_root':str(run) if run else None,'status':'NOT_STARTED','expected':spec['expected'],'cursor':0,'actual_input_tokens':0,'optimizer_step':0,'loss':None,'throughput':None,'elapsed':None,'ETA':None,'latest_DCP':None}
 if run and run.exists():
  active=matching_processes(run);row=last_row(run/'training.jsonl');result.update(status='RUNNING' if active else 'STOPPED',process_identities=active)
  if stage==1 and Path(spec['pointer']).exists():
   proc=read(spec['pointer']);identity=process_identity(proc.get('supervisor_pid'))
   if identity and spec['guard'] in identity['command'] and 'supervise' in identity['command']:
    active.append(identity);result.update(status='RUNNING',process_identities=active)
  if row:
   duration=row.get('iteration_seconds_excluding_checkpoint',0);tokens=sum(x.get('local_tokens',0) for x in row.get('rank_runtime',[]));rate=tokens/duration if duration else None
   result.update(cursor=row['record_cursor'],actual_input_tokens=row['cumulative_tokens'],optimizer_step=row['update_step'],loss=row.get('global_mean_loss'),throughput=rate,ETA=(spec['expected']['cumulative_tokens']-row['cumulative_tokens'])/rate if rate else None)
   result.update(cumulative_answer_positions=row.get('cumulative_answer_positions'),cumulative_context_positions=row.get('cumulative_context_positions'))
  if Path(spec['pointer']).exists():
   pointer=read(spec['pointer']);start=pointer.get('started',pointer.get('started_unix'))
   end=time.time()
   if (run/'launcher_exit.json').exists():end=read(run/'launcher_exit.json').get('ended',end)
   elif not active and row:end=row.get('logged_at_unix',end)
   result['elapsed']=max(0,end-start) if start else None
  cp=run/'checkpoints/latest.json'
  if cp.exists():result['latest_DCP']=read(cp)
  if (run/'complete.json').exists():result['completion']=read(run/'complete.json');result['status']='COMPLETION_PRESENT' if not active else 'FINAL_SHUTDOWN_PENDING'
 return result
def transition(state,next_state,authority,run,reason):
 previous=state.get('state','PRECHECK');require(next_state in STATES+['BLOCKED','FAILED'],'UNKNOWN_STATE')
 if next_state not in ['BLOCKED','FAILED'] and previous not in ['BLOCKED','FAILED']:
  require(STATES.index(next_state)>=STATES.index(previous),'STATE_REGRESSION')
 event={'timestamp':time.time(),'previous_state':previous,'next_state':next_state,'authority_hash':authority,'run_root':str(run) if run else None,'reason':reason}
 return {**state,'state':next_state,'updated_at':event['timestamp'],'history':state.get('history',[])+[event]}
