import os,time,json,subprocess,hashlib
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
controller=3742002
while True:
 try:os.kill(controller,0)
 except ProcessLookupError:break
 time.sleep(15)
receipts=json.loads((A/'RUN_RECEIPTS.json').read_text());assert len(receipts)==3 and all(x['status']=='VALID' for x in receipts)
records=[]
for script in ['test_analysis_cpu.py','audit_operators.py','analyze.py']:
 log=R/'logs'/('qwen3_4b_stage1_shared_boundary_v4_'+script+'.log');t=time.time();print('CPU_START',script,flush=True)
 with log.open('x') as f:ret=subprocess.run([str(R/'envs/ttt_runtime_v1/bin/python'),'-B',str(W/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({'script':script,'started_at':t,'ended_at':time.time(),'returncode':ret.returncode,'source_sha256':hashlib.sha256((W/script).read_bytes()).hexdigest(),'log':str(log)});(A/'CPU_RECEIPTS.json').write_text(json.dumps(records,indent=2)+'\n');assert ret.returncode==0,script
print('V4_ABC_CPU_COMPLETE',flush=True)
