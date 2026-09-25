import time,json,subprocess,os,sys
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';log=R/'logs/qwen3_4b_stage1_resume_v3_controller_attempt2.log'
while True:
 text=log.read_text()
 if 'STOP_FOR_CONCRETE_INPUT_OR_IMPLEMENTATION_FAILURE' in text:raise RuntimeError('GPU_CONTROLLER_STOPPED_REQUIRES_DIAGNOSIS')
 if 'REPLICATE_SCHEDULE_COMPLETE' in text:break
 time.sleep(30)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
for script in ['v3_input_audit.py','v3_metrics.py','v3_statistics.py','v3_reporting_evidence.py','v3_independent_auditor.py']:
 print('CPU_STAGE_START',script,time.time(),flush=True)
 target=R/'logs'/('qwen3_4b_stage1_resume_v3_'+script+'.log')
 with target.open('x') as f:subprocess.run([sys.executable,'-B',str(W/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('CPU_STAGE_PASS',script,time.time(),flush=True)
print('CPU_PIPELINE_COMPLETE',flush=True)
