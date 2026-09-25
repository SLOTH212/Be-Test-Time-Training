import time,json,subprocess,os,sys,hashlib
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';A=R/'audits/v3'
assert json.loads((A/'REPLICATE_INPUT_PARITY.json').read_text())['status']=='PASS'
opt=json.loads((A/'METRICS_CPU_EQUIVALENT_OPTIMIZATION.json').read_text())
assert hashlib.sha256((W/'v3_metrics_cached_tolerance.py').read_bytes()).hexdigest()==opt['optimized_source_sha256']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
receipts=[]
for script in ['v3_metrics_cached_tolerance.py','v3_statistics.py','v3_reporting_evidence.py','v3_independent_auditor.py','v3_reporting_independent_auditor.py']:
 item={'script':script,'source_sha256':hashlib.sha256((W/script).read_bytes()).hexdigest(),'started_at':time.time()};print('CPU_STAGE_START',script,item['started_at'],flush=True)
 target=R/'logs'/('qwen3_4b_stage1_resume_v3_'+script+'.log')
 with target.open('x') as f:
  proc=subprocess.Popen([sys.executable,'-B',str(W/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);item['pid']=proc.pid;proc.wait()
 item.update(returncode=proc.returncode,ended_at=time.time());receipts.append(item);(A/'CPU_ANALYSIS_OPTIMIZED_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 assert proc.returncode==0,('CPU_STAGE_FAILURE',script,proc.returncode)
 print('CPU_STAGE_PASS',script,item['ended_at'],flush=True)
print('CPU_PIPELINE_COMPLETE',flush=True)
