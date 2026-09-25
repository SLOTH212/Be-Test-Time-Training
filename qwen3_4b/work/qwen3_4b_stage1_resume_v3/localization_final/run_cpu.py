import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path('/path/to/ttt');L=R/'audits/v3/localization_final';N=R/'work/qwen3_4b_stage1_resume_v3/localization_final'
while True:
 p=L/'LOCALIZATION_RUN_RECEIPTS.json'
 if p.exists():
  try:receipts=json.loads(p.read_text())
  except ValueError:receipts=[]
  if any(x['status']!='VALID_COMPLETED_PHASE' for x in receipts):raise RuntimeError('LOCALIZATION_GPU_FAILURE')
  if len(receipts)==3:break
 time.sleep(2)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1');completed=[];running=[]
for script in ['analyze.py','audit_actual_operators.py']:
 log=R/'logs'/('qwen3_4b_stage1_resume_v3_localization_'+script+'.log');f=log.open('x');p=subprocess.Popen([sys.executable,'-B',str(N/script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT);running.append((p,f,{'script':script,'started_at':time.time(),'pid':p.pid,'source_sha256':hashlib.sha256((N/script).read_bytes()).hexdigest()}));print('CPU_START',script,flush=True)
for p,f,item in running:
 p.wait();f.close();item.update(ended_at=time.time(),returncode=p.returncode);completed.append(item)
(L/'CPU_RECEIPTS.json').write_text(json.dumps(completed,indent=2)+'\n');assert all(x['returncode']==0 for x in completed),completed
script='audit_final.py';item={'script':script,'started_at':time.time(),'source_sha256':hashlib.sha256((N/script).read_bytes()).hexdigest()}
with (R/'logs'/('qwen3_4b_stage1_resume_v3_localization_'+script+'.log')).open('x') as f:
 p=subprocess.Popen([sys.executable,'-B',str(N/script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT);p.wait()
item.update(ended_at=time.time(),returncode=p.returncode);completed.append(item);(L/'CPU_RECEIPTS.json').write_text(json.dumps(completed,indent=2)+'\n');assert p.returncode==0
print('LOCALIZATION_CPU_PIPELINE_COMPLETE',flush=True)
