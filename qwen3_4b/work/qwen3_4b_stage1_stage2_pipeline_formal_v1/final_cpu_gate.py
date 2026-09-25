import subprocess,json,time
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1';A=R/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1';python=str(R/'envs/ttt_runtime_v1/bin/python')
commands=[(['bash',str(R/'bin/launch_qwen3_4b_stage1_stage2_pipeline_formal_v1.sh')],'DRY_RUN.log'),([python,'-B',str(W/'independent_build_audit.py')],'INDEPENDENT_BUILD_AUDIT.log'),([python,'-B',str(W/'build_report.py')],'BUILD_REPORT.log')]
for cmd,log in commands:
 with (A/log).open('w') as file:result=subprocess.run(cmd,stdout=file,stderr=subprocess.STDOUT)
 if result.returncode:
  (A/'FINAL_CPU_GATE.json').write_text(json.dumps({'status':'FAIL','command':cmd,'log':str(A/log),'returncode':result.returncode,'time':time.time()},indent=2)+'\n');raise SystemExit(result.returncode)
(A/'FINAL_CPU_GATE.json').write_text(json.dumps({'status':'PASS','time':time.time(),'formal_training_started':False},indent=2)+'\n')
