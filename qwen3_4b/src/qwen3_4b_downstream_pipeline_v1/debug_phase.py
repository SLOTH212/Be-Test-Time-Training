#!/usr/bin/env python3
import os,sys,json,time,fcntl,subprocess,argparse
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_downstream_pipeline_debug_v1';S=R/'src/qwen3_4b_downstream_pipeline_v1'
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fixed','dynamic','reverse','reset','deletion','kp','r2'],required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
locks=[]
for gpu in [2,3]:
 f=(W/f'physical_gpu{gpu}.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
if a.phase=='fixed':assert json.loads((W/'sanity/workers/0/DONE.json').read_text())['status']=='PASS'
if a.phase=='dynamic':assert json.loads((W/'FIXED8_DEBUG_FINAL_AUTHORITY.json').read_text())['status']=='PASS'
if a.phase not in ['fixed','dynamic']:assert json.loads((W/'DYNAMIC_DEBUG_FINAL_AUTHORITY.json').read_text())['status']=='PASS'
ps=[];logs=[];started=time.time()
for worker in [0,1]:
 path=W/(a.phase+f'_worker{worker}.log');f=path.open('a');logs.append(f)
 cmd=[sys.executable,str(S/('debug_worker.py' if a.phase in ['fixed','dynamic'] else 'mechanism_worker.py')),'--phase' if a.phase in ['fixed','dynamic'] else '--experiment',a.phase,'--worker-id',str(worker),'--workers','2']
 ps.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy()))
(W/(a.phase+'_PHASE_STATE.json')).write_text(json.dumps(dict(phase=a.phase,pids=[p.pid for p in ps],started=started,status='RUNNING',DEBUG_ONLY=True,PAPER_RESULT=False)))
codes=[p.wait() for p in ps]
(W/(a.phase+'_PHASE_STATE.json')).write_text(json.dumps(dict(phase=a.phase,pids=[p.pid for p in ps],started=started,ended=time.time(),wall_seconds=time.time()-started,exit_codes=codes,status='PASS' if codes==[0,0] else 'FAIL',DEBUG_ONLY=True,PAPER_RESULT=False)))
raise SystemExit(0 if codes==[0,0] else 1)
