import os,sys,time,json,hashlib,subprocess,math
from pathlib import Path
R=Path('/path/to/ttt');A=R/'audits/v3';W=R/'work/qwen3_4b_stage1_resume_v3';RUN=R/'runs/qwen3_4b_stage1_resume_v3';RUN.mkdir(parents=True,exist_ok=True)
P=A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();H=sha(P);protocol=json.loads(P.read_text());assert H==P.with_suffix('.sha256').read_text().split()[0]
receipts=[]
def phase(replica,role):
 assert sha(P)==H
 for path,digest in protocol['input_file_sha256'].items():assert sha(path)==digest,('FROZEN_INPUT_CHANGED',path)
 out=RUN/(replica+'_pre' if role=='R_PRE' else replica);log=R/'logs'/('qwen3_4b_stage1_resume_v3_'+replica+'_'+role+'.log');assert not out.exists() and not log.exists()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3',WORLD_SIZE='2',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',V3_PROTOCOL_SHA256=H,V3_ROLE=role,V3_REPLICA=replica)
 for k in ['CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','STAGE1_CLOSURE_FAIL_DURING_SAVE_STEP']:env.pop(k,None)
 records={'U':10,'R_PRE':4,'R_POST':6}[role]
 cmd=[str(R/'envs/ttt_runtime_v1/bin/torchrun'),'--standalone','--nproc-per-node=2',str(W/'replicate_entry.py'),'--stage','1','--config',str(W/'original_replicate_config.yaml'),'--output-dir',str(out),'--max-records',str(records)]
 if role=='R_POST':cmd+=['--resume-from',str(RUN/(replica+'_pre')/'checkpoints')]
 start=time.time();receipt={'replica':replica,'role':role,'started_at':start,'protocol_sha256':H,'protocol_mtime':P.stat().st_mtime,'command':cmd,'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','V3_ROLE','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']},'output':str(out),'log':str(log)}
 assert receipt['protocol_mtime']<start
 (A/f'{replica}_{role}_START.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PHASE_START '+json.dumps(receipt),flush=True)
 with log.open('x') as file:p=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT,env=env,cwd=R);receipt['torchrun_pid']=p.pid;p.wait()
 receipt.update(returncode=p.returncode,ended_at=time.time());receipt['wall_seconds']=receipt['ended_at']-start
 norm=out/'parameter_norm_rank0.jsonl';receipt['optimizer_updates_observed']=len(norm.read_text().splitlines()) if norm.exists() else 0
 text=log.read_text();receipt['log_sha256']=sha(log)
 if p.returncode==0:
  c=json.loads((out/'complete.json').read_text());assert c['complete'] and c['record_cursor']==(4 if role=='R_PRE' else 10) and c['cumulative_tokens']==(115086 if role=='R_PRE' else 286544)
  for rank in [0,1]:assert json.loads((out/f'process_exit_rank{rank}.json').read_text())['status']=='PASS';assert json.loads((out/f'initial_parent_check_rank{rank}.json').read_text())['status']=='PASS'
  receipt['status']='VALID_COMPLETED_PHASE'
 else:
  is_resource=any(x in text for x in ['CUDA out of memory','Watchdog caught collective operation timeout','NCCL error','CUDA error: unknown error','driver shutting down'])
  receipt['status']='INVALID_RESOURCE_FAILURE' if is_resource else 'INVALID_IMPLEMENTATION_OR_INPUT_FAILURE'
 receipts.append(receipt);(A/f'{replica}_{role}_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');(A/'REPLICATE_CONTROLLER_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');print('PHASE_END '+json.dumps({k:v for k,v in receipt.items() if k not in ['command','environment']}),flush=True)
 if receipt['status']=='INVALID_IMPLEMENTATION_OR_INPUT_FAILURE':raise RuntimeError('STOP_FOR_CONCRETE_INPUT_OR_IMPLEMENTATION_FAILURE '+replica+' '+role)
 return receipt['status']=='VALID_COMPLETED_PHASE'
for replica in protocol['run_order']:
 if replica.startswith('U'):phase(replica,'U')
 elif phase(replica,'R_PRE'):phase(replica,'R_POST')
print('REPLICATE_SCHEDULE_COMPLETE '+json.dumps({'training_optimizer_updates':sum(x['optimizer_updates_observed'] for x in receipts),'two_gpu_wall_seconds':sum(x['wall_seconds'] for x in receipts),'protocol_sha256':H}),flush=True)
