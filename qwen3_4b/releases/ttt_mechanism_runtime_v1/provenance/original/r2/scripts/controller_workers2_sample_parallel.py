#!/usr/bin/env python3
"""Process-isolated two-worker controller for the frozen Exact-R2 run."""
import fcntl, json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
PY='/home/USER/conda_envs/ttt_phase_c_v1/bin/python'
WORKER=RUN/'scripts/exact_r2_sample_worker2.py'

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
 with open(tmp,'x') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def load(path):
 try:return json.load(open(path))
 except Exception:return {}
def alive(p): return p.poll() is None
def stop(ps):
 for p in ps:
  if alive(p): p.send_signal(signal.SIGTERM)
 deadline=time.time()+30
 while time.time()<deadline and any(alive(p) for p in ps):time.sleep(.2)
def main():
 lock=open(RUN/'status/controller.lock','w');
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: raise SystemExit('CONTROLLER_ALREADY_RUNNING')
 atomic(RUN/'status/controller.pid',{'pid':os.getpid(),'mode':'workers2_sample_parallel','started_at':now()})
 env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',EXACT_R2_WORKER2_CONTROLLER_PID=str(os.getpid()))
 logs=[];ps=[]
 try:
  for wid,cpus in [(0,'8-15'),(1,'16-23')]:
   lf=open(RUN/f'logs/worker{wid}_sample_parallel.log','a',buffering=1);logs.append(lf)
   ps.append(subprocess.Popen(['taskset','-c',cpus,PY,str(WORKER),'--worker-id',str(wid)],stdout=lf,stderr=subprocess.STDOUT,env=env,start_new_session=False))
  atomic(RUN/'status/workers2_controller.json',{'state':'RUNNING','pid':os.getpid(),'worker0_pid':ps[0].pid,'worker1_pid':ps[1].pid,'started_at':now()})
  while any(alive(p) for p in ps):
   s0=load(RUN/'status/worker0.json');s1=load(RUN/'status/worker1.json');a=s0.get('owned_sample');b=s1.get('owned_sample')
   if a and b and a==b:
    atomic(RUN/'errors/SAME_SAMPLE_DOUBLE_CLAIM.json',{'timestamp':now(),'sample_id':a,'worker0_pid':ps[0].pid,'worker1_pid':ps[1].pid});stop(ps);raise RuntimeError('SAME_SAMPLE_DOUBLE_CLAIM')
   if any(p.poll() not in (None,0) for p in ps): stop(ps);raise RuntimeError('WORKER_FAILURE '+str([p.poll() for p in ps]))
   time.sleep(.5)
  if any(p.returncode!=0 for p in ps):raise RuntimeError('WORKER_FAILURE '+str([p.returncode for p in ps]))
  atomic(RUN/'status/workers2_controller.json',{'state':'AGGREGATING','pid':os.getpid(),'worker0_pid':ps[0].pid,'worker1_pid':ps[1].pid,'updated_at':now()})
  subprocess.run(['taskset','-c','8-23',PY,str(RUN/'scripts/aggregate_exact_r2.py'),str(RUN)],check=True)
  subprocess.run([PY,str(RUN/'scripts/finalize_worker2_receipt.py')],check=True)
  atomic(RUN/'status/workers2_controller.json',{'state':'COMPLETE','pid':os.getpid(),'worker0_pid':ps[0].pid,'worker1_pid':ps[1].pid,'completed_at':now()})
 except BaseException as e:
  stop(ps);atomic(RUN/'status/workers2_controller.json',{'state':'FAILED','pid':os.getpid(),'error':repr(e),'updated_at':now()});raise
 finally:
  for f in logs:f.close()
if __name__=='__main__':main()
