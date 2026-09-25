#!/usr/bin/env python3
"""Continue only this already authorized bounded DEBUG pipeline, fail closed."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path('/path/to/ttt');S=R/'src/qwen3_4b_downstream_pipeline_v1';W=R/'work/qwen3_4b_downstream_pipeline_debug_v1';A=R/'audits/qwen3_4b_downstream_debug_v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(*args):
 print(json.dumps({'starting':list(map(str,args)),'time':time.time()}),flush=True);subprocess.run([sys.executable,*map(str,args)],check=True,env=os.environ.copy())
def hashes(root):return {str(p):sha(p) for p in Path(root).rglob('samples/*.json')}
def main():
 while read(W/'dynamic_PHASE_STATE.json')['status']=='RUNNING':time.sleep(10)
 assert read(W/'dynamic_PHASE_STATE.json')['status']=='PASS'
 run(S/'finalize_debug.py','dynamic');resume={}
 for phase in ['fixed','dynamic']:
  before=hashes(W/phase)
  for w in [0,1]:run(S/'debug_worker.py','--phase',phase,'--worker-id',str(w),'--workers','2')
  assert before==hashes(W/phase);resume[phase]={'status':'PASS','commits_unchanged':len(before),'completed_jobs_no_recompute':True}
  p=A/('FIXED8_DEBUG_AUDIT.json' if phase=='fixed' else 'DYNAMIC_DEBUG_AUDIT.json');x=read(p);x['actual_worker_resume']='PASS';x[phase.upper()+'_RESUME' if phase=='dynamic' else 'FIXED8_RESUME']='PASS';write(p,x)
 for ex in ['reverse','reset','deletion','kp','r2']:
  run(S/'debug_phase.py','--phase',ex)
  cl='REAL_CHAIN' if ex in ['reverse','r2'] else 'DEBUG_INTERFACE_FIXTURE';root=W/'mechanism'/cl/ex;before=hashes(root)
  for w in [0,1]:run(S/'mechanism_worker.py','--experiment',ex,'--worker-id',str(w),'--workers','2')
  assert before==hashes(root);resume[ex]={'status':'PASS','commits_unchanged':len(before),'completed_jobs_no_recompute':True}
 write(A/'RESUME_FINAL_AUDIT.json',{'status':'PASS','phases':resume,'DEBUG_ONLY':True,'PAPER_RESULT':False})
 started=time.time();ps=[]
 for w in [0,1]:
  f=(W/f'isolation_worker{w}.log').open('a');ps.append(subprocess.Popen([sys.executable,str(S/'state_isolation.py'),'--worker-id',str(w)],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy()))
 codes=[p.wait() for p in ps];write(W/'isolation_PHASE_STATE.json',dict(status='PASS' if codes==[0,0] else 'FAIL',exit_codes=codes,wall_seconds=time.time()-started,DEBUG_ONLY=True,PAPER_RESULT=False));assert codes==[0,0]
 p=R/'provenance/QWEN3_4B_4K_MECHANISM_INTEGRATION_CONTRACT_V1.json';x=read(p);x['status']='PASS';x['runtime_execution_evidence']=str(W/'mechanism');x['FORMAL_RESET_4B_32K_4K_ELIGIBILITY_STATUS']='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY';x['debug_fixture_resolves_formal_scientific_eligibility']=False;write(p,x)
 run(S/'independent_chain_audit.py');print('AUTHORIZED_DEBUG_PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
