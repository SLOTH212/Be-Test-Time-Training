import os,sys,json,time,hashlib,subprocess
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';N=W/'localization_final';A=R/'audits/v3';L=A/'localization_final';RUN=R/'runs/qwen3_4b_stage1_resume_v3/localization_final';RUN.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=L/'CONDITIONAL_LOCALIZATION_PLAN.json';H=sha(P);plan=json.loads(P.read_text());prior=json.loads((L/'PRIOR_EVIDENCE_IMMUTABILITY.json').read_text())['files'];audit=json.loads((L/'OBSERVER_READ_ONLY_AUDIT.json').read_text());assert audit['status']=='PASS' and audit['observer_sha256']==sha(N/'entry.py')==plan['observer_sha256']
receipts=[]
for replica,role,records in [('U_LOC','U',6),('R_LOC','R_PRE',4),('R_LOC','R_POST',2)]:
 assert sha(P)==H
 for p,h in prior.items():assert sha(p)==h,('IMMUTABLE_EVIDENCE_CHANGED',p)
 out=RUN/(replica+'_pre' if role=='R_PRE' else replica);log=R/'logs'/('qwen3_4b_stage1_resume_v3_localization_'+role+'.log');assert not out.exists() and not log.exists()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3',WORLD_SIZE='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',V3_PROTOCOL_SHA256=plan['frozen_protocol_sha256'],V3_LOCALIZATION_PLAN_SHA256=H,V3_ROLE=role,V3_REPLICA=replica)
 for k in ['CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','STAGE1_CLOSURE_FAIL_DURING_SAVE_STEP']:env.pop(k,None)
 cmd=[str(R/'envs/ttt_runtime_v1/bin/torchrun'),'--standalone','--nproc-per-node=2',str(N/'entry.py'),'--stage','1','--config',str(W/'original_replicate_config.yaml'),'--output-dir',str(out),'--max-records',str(records)]
 if role=='R_POST':cmd+=['--resume-from',str(RUN/'R_LOC_pre/checkpoints')]
 receipt={'replica':replica,'role':role,'started_at':time.time(),'command':cmd,'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']},'plan_sha256':H,'observer_sha256':sha(N/'entry.py'),'controller_sha256':sha(Path(__file__)),'output':str(out),'log':str(log)}
 assert P.stat().st_mtime<receipt['started_at'];(L/(role+'_START.json')).write_text(json.dumps(receipt,indent=2)+'\n');print('LOCALIZATION_PHASE_START',role,flush=True)
 with log.open('x') as f:
  proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);receipt['pid']=proc.pid;proc.wait()
 receipt.update(returncode=proc.returncode,ended_at=time.time());receipt['wall_seconds']=receipt['ended_at']-receipt['started_at'];receipt['log_sha256']=sha(log)
 pp=out/'parameter_norm_rank0.jsonl';receipt['optimizer_updates_observed']=len(pp.read_text().splitlines()) if pp.exists() else 0
 text=log.read_text();receipt['status']='VALID_COMPLETED_PHASE' if proc.returncode==0 else 'RESOURCE_RUNTIME_FAILURE' if any(x in text for x in ['out of memory','NCCL error','collective operation timeout','driver shutting down']) else 'OBSERVER_OR_IMPLEMENTATION_FAILURE'
 if proc.returncode==0:
  c=json.loads((out/'complete.json').read_text());assert c['complete'] and c['record_cursor']==(4 if role=='R_PRE' else 6)
  for rank in (0,1):assert json.loads((out/f'process_exit_rank{rank}.json').read_text())['status']=='PASS'
 receipts.append(receipt);(L/(role+'_RECEIPT.json')).write_text(json.dumps(receipt,indent=2)+'\n');(L/'LOCALIZATION_RUN_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');print('LOCALIZATION_PHASE_END',role,receipt['status'],receipt['wall_seconds'],flush=True)
 if proc.returncode:raise RuntimeError(receipt['status'])
assert sum(x['optimizer_updates_observed'] for x in receipts)==6
print('LOCALIZATION_GPU_PAIR_COMPLETE',flush=True)
