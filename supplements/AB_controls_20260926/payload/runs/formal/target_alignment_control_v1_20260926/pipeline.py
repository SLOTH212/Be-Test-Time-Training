#!/usr/bin/env python3
import os,sys,subprocess,json,time,signal
from pathlib import Path
os.sched_setaffinity(0,set(range(8,24)))
R=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def call(args,log):
 with (R/log).open('a') as f:
  p=subprocess.Popen([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,cwd=R)
  rc=p.wait()
 if rc:raise RuntimeError(f'{args} exited {rc}; see {log}')
def status(phase,**kw):
 tmp=R/'pipeline_status.tmp';tmp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw),indent=2));tmp.replace(R/'pipeline_status.json')
children=[]
try:
 status('SMOKE_RUNNING')
 call([R/'run.py','smoke','0'],'smoke.log')
 assert json.loads((R/'SMOKE_RESULT.json').read_text())['status']=='PASS'
 status('FORMAL_RUNNING')
 logs=[]
 for w in range(2):
  f=(R/f'formal_{w}.log').open('a');logs.append(f)
  children.append(subprocess.Popen([sys.executable,str(R/'run.py'),'formal',str(w)],stdout=f,stderr=subprocess.STDOUT,cwd=R))
 while any(p.poll() is None for p in children):
  if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Formal worker failed; stopping sibling')
  time.sleep(2)
 assert all(p.returncode==0 for p in children)
 for f in logs:f.close()
 status('STATISTICS_RUNNING')
 call([R/'run.py','integrity'],'integrity.log')
 call([R/'summarize.py'],'statistics.log')
 summary=json.loads((R/'bootstrap_summary.json').read_text());assert summary['status']=='PASS',summary
 status('COMPLETE',result='PASS')
except BaseException as e:
 for p in children:
  if p.poll() is None:p.terminate()
 for p in children:
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 status('FAILED',error=str(e))
 try:call([R/'run.py','integrity'],'integrity.log');call([R/'summarize.py'],'statistics.log')
 except Exception:pass
 raise
