import os,sys,json,time,hashlib,subprocess
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4_1';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4_1';RUN=R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4_1';RUN.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=A/'MISSING_CAPTURE_CLOSURE_PROTOCOL_V4_1.json';H=sha(P);plan=json.loads(P.read_text());prior=json.loads((A/'PRIOR_AUTHORITIES_IMMUTABLE.json').read_text())['files'];assert sha(W/'entry.py')==plan['observer_sha256']
freeze=json.loads((A/'PROTOCOL_FREEZE_RECEIPT.json').read_text());assert freeze['sha256']==H
branches=['A','B1','B2','B3']
receipts=json.loads((A/'RUN_RECEIPTS.json').read_text()) if (A/'RUN_RECEIPTS.json').exists() else []
def dcp_identity():
 root=RUN/'A/checkpoints';latest=json.loads((root/'latest.json').read_text());manifest=root/latest['slot']/'manifest.json';assert sha(manifest)==latest['manifest_sha256']
 content=json.loads(manifest.read_text());return {'root':str(root),'latest_sha256':sha(root/'latest.json'),'manifest_sha256':sha(manifest),'files':content['files'],'progress':content['progress']}
for branch in branches:
 assert sha(P)==H and sha(W/'entry.py')==plan['observer_sha256'] and sha(plan['config'])==plan['config_sha256']
 for p,h in prior.items():assert sha(p)==h,('PRIOR_CHANGED',p)
 if branch!='A':
  identity=dcp_identity();assert identity==json.loads((A/'SHARED_DCP_IDENTITY.json').read_text())
 out=RUN/branch;log=R/'logs'/('qwen3_4b_stage1_shared_boundary_v4_1_'+branch+'.log');assert not out.exists() and not log.exists()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3',WORLD_SIZE='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',V3_PROTOCOL_SHA256=plan['prior_v3_protocol_sha256'],V3_LOCALIZATION_PLAN_SHA256=sha(R/'audits/v3/localization_final/CONDITIONAL_LOCALIZATION_PLAN.json'),V3_ROLE='U' if branch=='A' else 'R_POST',V3_REPLICA=branch,V4_BRANCH=branch,V4_PROTOCOL_SHA256=H)
 for k in ['CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED','STAGE1_CLOSURE_FAIL_DURING_SAVE_STEP']:env.pop(k,None)
 cmd=[str(R/'envs/ttt_runtime_v1/bin/torchrun'),'--standalone','--nproc-per-node=2',str(W/'entry.py'),'--stage','1','--config',plan['config'],'--output-dir',str(out),'--max-records','6' if branch=='A' else '2']
 if branch!='A':cmd+=['--resume-from',str(RUN/'A/checkpoints')]
 receipt={'DEBUG_ONLY':True,'branch':branch,'started_at':time.time(),'command':cmd,'environment':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED']},'protocol_sha256':H,'observer_sha256':sha(W/'entry.py'),'controller_sha256':sha(Path(__file__)),'output':str(out),'log':str(log),'load_identity':None if branch=='A' else identity}
 assert freeze['frozen_at']<receipt['started_at'];(A/(branch+'_START.json')).write_text(json.dumps(receipt,indent=2)+'\n');print('V4_BRANCH_START',branch,flush=True)
 with log.open('x') as f:
  proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);receipt['pid']=proc.pid;proc.wait()
 receipt.update(returncode=proc.returncode,ended_at=time.time());receipt['wall_seconds']=receipt['ended_at']-receipt['started_at'];receipt['log_sha256']=sha(log)
 pp=out/'parameter_norm_rank0.jsonl';receipt['optimizer_updates_observed']=len(pp.read_text().splitlines()) if pp.exists() else 0
 receipt['status']='VALID' if proc.returncode==0 else 'EXECUTION_FAILURE'
 if proc.returncode==0:
  c=json.loads((out/'complete.json').read_text());assert c['complete'] and c['record_cursor']==6
  assert receipt['optimizer_updates_observed']==(3 if branch=='A' else 1)
  for rank in (0,1):
   assert json.loads((out/f'process_exit_rank{rank}.json').read_text())['status']=='PASS'
   assert json.loads((out/f'final_rank{rank}.json').read_text())['status']=='PASS'
  if branch=='A':
   for rank in (0,1):assert json.loads((out/f'boundary_save_rank{rank}.json').read_text())['status']=='PASS'
   (A/'SHARED_DCP_IDENTITY.json').write_text(json.dumps(dcp_identity(),indent=2)+'\n')
  else:assert dcp_identity()==identity
 receipts.append(receipt);(A/(branch+'_RECEIPT.json')).write_text(json.dumps(receipt,indent=2)+'\n');(A/'RUN_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');print('V4_BRANCH_END',branch,receipt['status'],receipt['wall_seconds'],flush=True)
 if proc.returncode:raise RuntimeError('V4_BRANCH_FAILED_STOP')
assert sum(x['optimizer_updates_observed'] for x in receipts)==6
print('V4_GPU_SCHEDULE_COMPLETE',branches,flush=True)
