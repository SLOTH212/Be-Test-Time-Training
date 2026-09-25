#!/usr/bin/env python3
import argparse,json,hashlib,os,subprocess,sys,time
from pathlib import Path
R=Path('/path/to/ttt')
CONFIG=R/'configs/qwen3_4b_stage2_30m_32k_gpu45_formal_v1.yaml'
PREFLIGHT=R/'provenance/QWEN3_4B_STAGE2_30M_32K_GPU23_DEBUG_PREFLIGHT_V1.json'
WORKER=R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def owned(p):
 p=Path(p).resolve()
 if not p.is_relative_to(R.resolve()):raise RuntimeError('OUTPUT_OR_PARENT_OUTSIDE_AUTHORIZED_ROOT')
 return p
def load(p):return json.loads(Path(p).read_text())
def parent_contract(path,debug_test=False):
 path=owned(path);a=load(path)
 if debug_test:
  if a.get('PARENT_CLASS')!='DEBUG_ONLY' or a.get('debug_integrity')!='PASS':raise RuntimeError('DEBUG_PARENT_INVALID')
 else:
  if a.get('PARENT_CLASS')=='DEBUG_ONLY' or a.get('DEBUG_ONLY'):raise RuntimeError('DEBUG_PARENT_FORBIDDEN_FORMAL')
  if a.get('STAGE1_FINAL_INTEGRITY')!='PASS' or a.get('stage1_completed') is not True:raise RuntimeError('STAGE1_FINAL_REQUIRED')
  if a.get('record_cursor')!=38335 or a.get('cumulative_tokens')!=989996971:raise RuntimeError('STAGE1_COMPLETION_COUNTER_MISMATCH')
 for k in ('checkpoint_path','checkpoint_sha256','manifest_sha256','config_sha256','tokenizer_sha256','model_identity','ttt_layers','context_length'):
  if not a.get(k):raise RuntimeError('PARENT_REQUIRED_FIELD_'+k)
 if a['model_identity']!='Qwen3-4B' or a['ttt_layers']!=[0,6,12,18,24,30] or a['context_length']!=32768:raise RuntimeError('PARENT_MODEL_CONTRACT')
 cp=owned(a['checkpoint_path']);m=load(cp/'manifest.json')
 meta=m.get('metadata',{})
 for k in ('config_sha256','tokenizer_sha256','model_identity','ttt_layers','context_length'):
  if meta.get(k)!=a[k]:raise RuntimeError('PARENT_METADATA_MISMATCH_'+k)
 if m.get('format')!='gated_ntp_stage1_model_only_dcp_v1' or not m.get('complete'):raise RuntimeError('PARENT_MODEL_ONLY_FORMAT')
 if not debug_test:
  meta=m.get('metadata',{})
  if meta.get('DEBUG_ONLY') or meta.get('PARENT_CLASS')=='DEBUG_ONLY':raise RuntimeError('ACTUAL_PARENT_IS_DEBUG_ONLY')
  if meta.get('record_cursor',meta.get('progress',{}).get('record_cursor'))!=38335:raise RuntimeError('ACTUAL_PARENT_NOT_STAGE1_FINAL')
 if sha(cp/'manifest.json')!=a['manifest_sha256'] or sha(cp/'model.pt')!=a['checkpoint_sha256']:raise RuntimeError('PARENT_HASH_MISMATCH')
 if m['files'].get('model.pt')!=a['checkpoint_sha256']:raise RuntimeError('PARENT_MANIFEST_MODEL_HASH')
 if sha(R/'shared/models/Qwen3-4B-Base/tokenizer.json')!=a['tokenizer_sha256']:raise RuntimeError('PARENT_TOKENIZER_MISMATCH')
 return a
def data_contract():
 d=R/'shared/datasets'
 expected={'authority/STAGE2_DATASET_AUTHORITY.json':'e423c322602c5124acc839470079cfe23559b43c0a29eb9042cf1a3a4ee6ddd0','manifests/FINAL_30M_QA_CONTENT_ORDER.jsonl':'8ecf2aa5a6ba1fe8c6a6654c588d6d8810dcd9c38d8f12989d4ee44fc319ca43','manifests/FINAL_PACKED_RECORDS.jsonl':'4b9a4610effaba35a616c7622c248ee79db18bbd50f1a82a0e5c0d0326c13898','authority/stage2_qa_replay_30m_32k_qwen3_v2.yaml':'3aec55e62ee30dece2d79094989f8638f5fd754730f9a0aa330eb0ba4109e383'}
 for n,h in expected.items():
  if sha(d/n)!=h:raise RuntimeError('STAGE2_DATA_IDENTITY_MISMATCH '+n)
 a=load(d/'authority/STAGE2_DATASET_AUTHORITY.json')
 if a['dataset_id']!='QA_REPLAY_30M_32K_QWEN3_STAGE2_V2':raise RuntimeError('STAGE2_DATASET_ID')
 return a
def check(parent):
 sys.path.insert(0,'/path/to/ttt/work/qwen3_4b_stage1_stage2_pipeline_formal_v1')
 from gpu45_resource_loader import resource_contract
 resource_contract()
 a=parent_contract(parent);data_contract()
 subprocess.run(['bash',str(R/'bin/check_shared_assets.sh')],check=True)
 cfg=load(CONFIG)
 if sha(CONFIG)!=CONFIG.with_suffix('.sha256').read_text().split()[0]:raise RuntimeError('FORMAL_CONFIG_HASH_MISMATCH')
 if sha(WORKER)!=cfg['input_identity']['worker_sha256']:raise RuntimeError('WORKER_HASH_MISMATCH')
 for p,h in cfg['data_file_sha256'].items():
  if sha(p)!=h:raise RuntimeError('RAW_DATA_HASH_MISMATCH '+p)
 for key,p in [('stage2_checkpoint_adapter_sha256',R/'src/training_runtime/code/lib/stage2_checkpoint_runtime.py'),('stage2_entry_sha256',R/'bin/stage2_runtime_entry_gpu45_pipeline_v1.py')]:
  if sha(p)!=cfg['input_identity'][key]:raise RuntimeError('RUNTIME_ADAPTER_HASH_MISMATCH '+str(p))
 auth=load(PREFLIGHT)
 if auth.get('QWEN3_4B_STAGE2_PREFLIGHT_STATUS') not in ('PASS','PASS_WITH_WARNING') or auth.get('SAFE_TO_RUN_FORMAL_STAGE2_AFTER_VALID_STAGE1_PARENT')!='YES':raise RuntimeError('STAGE2_PREFLIGHT_NOT_CLEARED')
 for p,h in auth.get('file_sha256',{}).items():
  if sha(p)!=h:raise RuntimeError('BOUND_RUNTIME_HASH_MISMATCH '+p)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5':raise RuntimeError('GPU_ALLOWLIST')
 gpu=subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 for line in gpu.splitlines():
  idx,free=map(int,line.split(','))
  if idx not in (4,5) or free<48000:raise RuntimeError('GPU_MEMORY_CAPACITY_NOT_AVAILABLE')
 return a,cfg
def status():
 pointer=R/'runs/qwen3_4b_stage2_formal_current.json'
 out={'formal_run_created':pointer.exists(),'preflight':load(PREFLIGHT).get('QWEN3_4B_STAGE2_PREFLIGHT_STATUS'),'process_state':'NOT_STARTED','parent_stage1_authority':None,'stage2_cursor':0,'actual_input_tokens':0,'supervised_answer_tokens':0,'optimizer_step':0,'latest_loss':None,'throughput':None,'latest_dcp':None,'elapsed_seconds':None,'eta_seconds':None,'errors':[]}
 if pointer.exists():
  a=load(pointer);root=owned(a['run_root']);out.update(a);out['process_alive']=Path('/proc') .joinpath(str(a['pid'])).exists()
  log=root/'training.jsonl'
  if log.exists():
   rows=log.read_text().splitlines();last=json.loads(rows[-1]);out['latest']=last
   elapsed=time.time()-a['started_unix'];rate=last['cumulative_tokens']/elapsed if elapsed>0 else None
   out.update({'process_state':'RUNNING' if out['process_alive'] else 'STOPPED','parent_stage1_authority':a['parent_authority'],'stage2_cursor':last['record_cursor'],'actual_input_tokens':last['cumulative_tokens'],'supervised_answer_tokens':last.get('cumulative_answer_positions'),'optimizer_step':last['update_step'],'latest_loss':last.get('global_mean_loss'),'throughput':rate,'elapsed_seconds':elapsed,'eta_seconds':(30000000-last['cumulative_tokens'])/rate if rate else None})
  cp=root/'checkpoints/latest.json'
  if cp.exists():out['latest_dcp']=load(cp)
  done=root/'complete.json'
  if done.exists():
   completed=load(done);out['completion']=completed
   out['process_state']='COMPLETED' if completed.get('complete') else 'INCOMPLETE'
  launcher=root/'launcher.log'
  if launcher.exists():out['errors']=[line for line in launcher.read_text(errors='replace').splitlines()[-200:] if any(tag in line for tag in ('Traceback','RuntimeError','ERROR','OutOfMemoryError'))][-20:]
 out['gpu23']=subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,power.draw','--format=csv'],text=True)
 print(json.dumps(out,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','launch','status']);ap.add_argument('--parent-authority');args=ap.parse_args()
 if args.action=='status':return status()
 if not args.parent_authority:raise RuntimeError('EXPLICIT_PARENT_AUTHORITY_REQUIRED')
 a,cfg=check(args.parent_authority)
 if args.action=='check':print('FORMAL_CONTRACT_CHECK_PASS');return
 root=R/'runs'/('qwen3_4b_stage2_formal_'+time.strftime('%Y%m%dT%H%M%S'));root.mkdir(exist_ok=False)
 # Parent-specific run identity is bound separately from frozen scientific config.
 runtime_cfg=dict(cfg);runtime_cfg['input_identity']={**cfg['input_identity'],'parent_authority_sha256':sha(args.parent_authority),'parent_checkpoint_sha256':a['checkpoint_sha256']}
 run_cfg=root/'resolved_config.yaml';run_cfg.write_text(json.dumps(runtime_cfg,indent=2)+'\n')
 log=(root/'launcher.log').open('x')
 p=subprocess.Popen(['torchrun','--standalone','--nproc-per-node=2',str(R/'bin/stage2_runtime_entry_gpu45_pipeline_v1.py'),'--stage','2','--config',str(run_cfg),'--output-dir',str(root),'--parent-artifact',a['checkpoint_path']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 pointer={'run_root':str(root),'pid':p.pid,'parent_authority':str(args.parent_authority),'started_unix':time.time(),'FORMAL_STAGE2_TRAINING_STARTED':True}
 (R/'runs/qwen3_4b_stage2_formal_current.json').write_text(json.dumps(pointer,indent=2)+'\n');print(json.dumps(pointer))
if __name__=='__main__':main()
