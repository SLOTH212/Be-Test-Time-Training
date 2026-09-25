"""Formal Llama pipeline contracts; no training mathematics or CUDA imports."""
import os,json,hashlib,fcntl,contextlib,time,subprocess,shutil,math
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/llama31_8b_stage1_stage2_pipeline_formal_v1'
CONFIG=R/'configs/llama31_8b_stage1_stage2_pipeline_formal_v1.yaml'
SOURCE=R/'provenance/LLAMA31_8B_STAGE1_STAGE2_PIPELINE_SOURCE_AUTHORITY_V1.json'
STATES=['PRECHECK','STAGE1_READY','STAGE1_RUNNING','STAGE1_FINALIZING','STAGE1_FINAL_FROZEN','STAGE2_READY','STAGE2_RUNNING','STAGE2_FINALIZING','STAGE2_FINAL_FROZEN','PIPELINE_COMPLETE']
EXPECTED={1:{'record_cursor':17015,'cumulative_tokens':478374652,'update_step':4254},2:{'record_cursor':457,'cumulative_tokens':14390540,'update_step':115,'cumulative_answer_positions':97619,'cumulative_context_positions':14292464}}
SPLIT={'train_records':17015,'train_input_tokens':478374652,'validation_records':185,'validation_input_tokens':4850937,'total_package_records':17200,'total_package_input_tokens':483225589,'optimizer_steps':4254,'validation_used_for_training':False,'train_validation_separation_verified':True}
def require(v,msg):
 if not v:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);require(p.resolve().is_relative_to(R),'WRITE_OUTSIDE_ROOT');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def config():return read(CONFIG)
def hashes(files):
 for p,h in files.items():require(sha(p)==h,'BOUND_HASH_MISMATCH '+p)
def identity(pid):
 try:
  p=Path('/proc')/str(int(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split()
  if s[0]=='Z':return None
  return {'pid':int(pid),'start_ticks':int(s[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
 except (OSError,ValueError,TypeError):return None
def alive(p):return bool(p and identity(p.get('pid'))==p)
@contextlib.contextmanager
def lock(p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a')
 try:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('EXCLUSIVE_LOCK_ALREADY_HELD')
  yield f
 finally:f.close()
def source():
 a=read(SOURCE);require(sha(SOURCE)==SOURCE.with_suffix('.sha256').read_text().split()[0],'SOURCE_AUTHORITY_HASH');hashes(a['files']);return a
def split_contract(x):
 for k,v in SPLIT.items():require(x.get(k)==v,'STAGE1_SPLIT_'+k)
def dataset_contract(a1,a2):
 require(a1['status']=='PASS' and a1['packed_record_counts']=={'train':17015,'validation':185} and a1['actual_llama_tokens']=={'train':478374652,'validation':4850937},'STAGE1_DATA_AUTHORITY')
 require(a2['status']=='PASS' and all(a2['packing'][k]==v for k,v in {'records':457,'input_tokens':14390540,'qa_targets':19397,'supervised_answer_tokens':97619}.items()),'STAGE2_DATA_AUTHORITY')
def cfg_contract(c,stage):
 require(c['record_cursor_final']==EXPECTED[stage]['record_cursor'] and c['train_tokens']==EXPECTED[stage]['cumulative_tokens'] and c['expected_optimizer_steps']==EXPECTED[stage]['update_step'],'TRAIN_COMPLETION_COUNTS')
 require(c['world_size']==4 and c['global_batch_size']==4 and c['micro_batch_size']==c['gradient_accumulation']==1 and c['physical_gpu_allowlist']=='0,1,2,3','GPU_BATCH_CONTRACT')
 require(c['context_length']==32768 and c['ttt_chunk_size']==4096 and c['ttt_layers']==[0,6,12,18,24,30],'SCIENCE_CONTRACT')
 require((c['optimizer'],c['outer_lr'],c['betas'],c['eps'],c['weight_decay'],c['outer_grad_clip'],c['scheduler'],c['precision'])==('AdamW',5e-6,[.9,.95],1e-8,.1,1.,'constant','bf16'),'OPTIMIZER_CONTRACT')
 require(read(c['train_data'])['splits']==['train'],'VALIDATION_IN_TRAIN_LOADER')
 if stage==1:split_contract(c)
 else:require(c['context_group_weight']==.1,'STAGE2_WEIGHTED_LOSS')
def storage(c):
 free=shutil.disk_usage(R).free;require(free>=c['storage_required_bytes'],'INSUFFICIENT_STORAGE');return {'free_bytes':free,'required_bytes':c['storage_required_bytes']}
def gpu():
 expected=['GPU-c7982600-dea0-9d54-16b8-76db4288663e','GPU-8459f05d-4d22-5b39-cc1f-e25137d49389','GPU-fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','GPU-351e23d4-250b-6a70-2c66-d2fb0221a25a']
 text=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True);rows=[x.split(',') for x in text.strip().splitlines()]
 require(len(rows)==4 and all(int(r[0])==i and r[1].strip()==expected[i] and float(r[2])>=47500 for i,r in enumerate(rows)),'GPU_MAPPING_OR_FREE_MEMORY');return text
def preflight(c,live=False):
 import yaml
 source();identity_contract(c);hashes(c['bindings']);dataset_contract(read(c['stage1']['data_authority']),read(c['stage2']['data_authority']))
 for stage in [1,2]:cfg_contract(yaml.safe_load(Path(c[f'stage{stage}']['config']).read_text()),stage)
 require(c['downstream_auto_start'] is False and c['final_long_term_model_artifacts']==2,'PIPELINE_SCOPE')
 checks={'storage':storage(c)}
 if live:checks['gpu']=gpu()
 return checks
def identity_contract(c):
 require(c['model_revision']=='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b','MODEL_REVISION')
 require(c['port_sha256']=='3f1ca6c6cee8f8601fd5bfca396a067fe1876cbba3fbabecb273e75e19fd54bc','PORT_AUTHORITY')
 for name,h in [('llama31_stage1_500m_32k_v1.tar.gz','c7189fb46adad303b97c7a3cee36059bb201e9b7b5fa2cbf3772a4535c841680'),('llama31_stage2_15m_32k_v1.tar.gz','6bb877e4ece2819fb063581ff276c305574822a3c55a1dcfa31ed8cc79030227')]:require(c['bindings'].get(str(R/'packages'/name))==h,'DATA_PACKAGE_AUTHORITY')
def last(run):
 p=Path(run)/'training.jsonl'
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-(4<<20)));lines=f.read().splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
 return None
def trainer_snapshot(stage,c):
 run=Path(c[f'stage{stage}']['run_root']);p=run/'process.json';proc=read(p) if p.exists() else {};active=[x for x in [proc.get('manager'),proc.get('torchrun')]+proc.get('workers',[]) if alive(x)]
 return {'run_root':str(run),'active':active,'process':proc,'row':last(run),'complete':(run/'complete.json').exists(),'exit':read(run/'launcher_exit.json') if (run/'launcher_exit.json').exists() else None}
def health(row,stage,c):
 require(math.isfinite(row['global_mean_loss']) and math.isfinite(row['grad_norm_before_clip']),'NONFINITE_LOSS_OR_GRADIENT')
 index=read(c[f'stage{stage}']['record_index']);cursor=row['record_cursor'];step=row['update_step'];require(cursor==min(step*4,len(index)),'STEP_CURSOR')
 require(row['cumulative_tokens']==sum(x['tokens'] for x in index[:cursor]),'GLOBAL_TOKEN_ACCOUNTING')
 seen=[]
 for r in row['rank_runtime']:
  rank=r['rank'];require(rank in range(4) and r['physical_gpu']==rank and r['weights_finite'] is True,'GPU_OR_FINITE_WEIGHTS')
  want=[x for i,x in enumerate(index[max(0,(step-1)*4):cursor],start=(step-1)*4) if i%4==rank]
  require(r['sample_ids']==[x['id'] for x in want],'RECORD_OWNERSHIP_OR_VALIDATION_CONTAMINATION');require(r['local_tokens']==sum(x['tokens'] for x in want),'RANK_TOKEN_ACCOUNTING');seen+=r['sample_ids']
 require(len(seen)==len(set(seen)) and len(row['rank_runtime'])==4,'DUPLICATE_REAL_RECORD_OR_RANK')
def dcp(run,expected):
 root=Path(run)/'checkpoints';latest=read(root/'latest.json');require(latest['slot'] in ['slot_A','slot_B'],'DCP_SLOT');slot=root/latest['slot'];require(sha(slot/'manifest.json')==latest['manifest_sha256'],'DCP_MANIFEST');a=read(slot/'manifest.json');require(a['complete'],'DCP_INCOMPLETE')
 for k,v in expected.items():require(a['progress'].get(k)==v,'DCP_COUNTER_'+k)
 hashes({str(slot/p):h for p,h in a['files'].items()});return slot,a
def completion(stage,c):
 snap=trainer_snapshot(stage,c);run=Path(snap['run_root']);require(not snap['active'],'TRAINER_ACTIVE');require(snap['exit'] and snap['exit']['exit_code']==0,'TRAINER_EXIT');done=read(run/'complete.json');require(done['complete'] and done['stage']==stage,'COMPLETION_MARKER');row=snap['row'];health(row,stage,c)
 for k,v in EXPECTED[stage].items():require(row.get(k)==v,'FINAL_COUNTER_'+k)
 log=(run/'launcher.log').read_text(errors='replace');require('FINAL_BARRIER_PASS' in log and all(f'RANK_{i}_SHUTDOWN_PASS' in log for i in range(4)),'CLEAN_SHUTDOWN');require(not any(x in log for x in ['Traceback (most recent call last)','OutOfMemoryError','NCCL error']),'TRAINER_ERRORS')
 slot,manifest=dcp(run,EXPECTED[stage]);return {'slot':str(slot),'manifest_sha256':sha(slot/'manifest.json'),'progress':manifest['progress'],'run_root':str(run),'config_sha256':sha(c[f'stage{stage}']['config'])}
def validate_final(root,stage,c):
 root=Path(root);a=read(root/f'STAGE{stage}_FINAL_AUTHORITY.json');require(a.get('artifact_class')==f'STAGE{stage}_FINAL' and a.get('DEBUG_ONLY') is False and a.get('immutable') is True,'FINAL_CLASS')
 require(a.get('final_root')==str(root) and a.get('integrity')=='PASS','FINAL_IDENTITY')
 for k,v in EXPECTED[stage].items():require(a.get(k)==v,'FINAL_COUNTER_'+k)
 if stage==1:split_contract(a)
 require(a.get('source_authority_sha256')==sha(SOURCE),'FINAL_SOURCE_PARENT');require(a.get('data_authority_sha256')==sha(c[f'stage{stage}']['data_authority']),'FINAL_DATA_PARENT')
 require({'config.json','tokenizer.json','SHA256SUMS.txt'}.issubset(a['files']),'FINAL_STANDALONE')
 require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(a['files'])|{f'STAGE{stage}_FINAL_AUTHORITY.json'},'UNBOUND_FINAL_FILES')
 for p,h in a['files'].items():require(not (root/p).is_symlink() and (root/p).resolve().is_relative_to(root) and sha(root/p)==h,'FINAL_FILE_HASH')
 if stage==2:require(a.get('inference_ready')=='PASS' and a.get('stage1_final_authority_sha256')==sha(Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json'),'STAGE2_FINAL_PARENT')
 return a
