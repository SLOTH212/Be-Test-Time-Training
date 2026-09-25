import os,time,json,subprocess,hashlib
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
while True:
 try:os.kill(3763919,0)
 except ProcessLookupError:break
 time.sleep(15)
rs=json.loads((A/'RUN_RECEIPTS.json').read_text());assert [x['branch'] for x in rs]==['A','B1','B2','B3'] and all(x['status']=='VALID' for x in rs)
receipts=[]
for script in ['audit_operators.py','analyze.py']:
 t=time.time();log=R/'logs'/('qwen3_4b_stage1_shared_boundary_v4_B3_'+script+'.log');print('B3_CPU_START',script,flush=True)
 with log.open('x') as f:p=subprocess.run([str(R/'envs/ttt_runtime_v1/bin/python'),'-B',str(W/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 receipts.append({'DEBUG_ONLY':True,'script':script,'started_at':t,'ended_at':time.time(),'returncode':p.returncode,'source_sha256':hashlib.sha256((W/script).read_bytes()).hexdigest(),'log':str(log)});(A/'CPU_B3_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');assert p.returncode==0
print('V4_B3_CPU_COMPLETE',flush=True)
