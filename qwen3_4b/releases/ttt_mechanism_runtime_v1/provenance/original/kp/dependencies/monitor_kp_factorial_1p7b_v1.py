#!/usr/bin/env python3
import json, os, subprocess, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/kp_factorial_1p7b_v1/run_20260905T235503+0800')
TMUX='/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/tools/tmux';SESSION='kp_factorial_1p7b_v1'
def load(p):
 try:return json.loads(Path(p).read_text())
 except Exception:return {}
def alive(pid):return bool(pid and Path(f'/proc/{pid}').exists())
def proc(pattern):
 out=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if pattern in cmd and 'monitor_kp_factorial' not in cmd:out.append(int(p.name))
  except Exception:pass
 return sorted(out)
status=load(RUN/'status/pipeline_status.json');files=sorted((RUN/'branches').glob('*.json'));valid=[];temps=list((RUN/'branches').glob('.*.tmp'))
for p in files:
 x=load(p)
 if x.get('completed') is True and x.get('status')=='SUCCESS':valid.append(x)
counts=Counter(x.get('cell') for x in valid);by=defaultdict(set)
for x in valid:by[x.get('canonical_sample_id')].add(x.get('cell'))
four=sum(v=={'F00','F10','F01','F11'} for v in by.values());dups=len(valid)-len({x.get('branch_id') for x in valid})
mt=sorted(p.stat().st_mtime for p in files);last=max(mt) if mt else None
rate=0.0
if len(mt)>1:
 q=mt[-min(20,len(mt)):];rate=(len(q)-1)/max(q[-1]-q[0],1e-9)*3600
eta='UNKNOWN' if rate<=0 else f'{(668-len(valid))/rate:.3f}h'
try:tmux_ok=subprocess.run([TMUX,'has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
except Exception:tmux_ok=False
cp=proc('run_kp_controller.sh');wp=proc('kp_worker.py');worker=wp[0] if wp else None
aff=''
if worker:
 try:aff=subprocess.check_output(['taskset','-pc',str(worker)],text=True).strip().split(':',1)[-1].strip()
 except Exception:aff='UNKNOWN'
gpu=['UNKNOWN']*7
try:
 g=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(', ');gpu=g
except Exception:pass
log=''
for p in [RUN/'logs/controller.log']:
 try:log+=p.read_text(errors='replace')[-200000:]
 except Exception:pass
errs=list((RUN/'errors').glob('*.json'));receipt=RUN/'reports/KP_FACTORIAL_1P7B_FORMAL_COMPLETION_RECEIPT.json'
risk='NONE'
if 'out of memory' in log.lower():risk='OOM'
elif 'Traceback' in log:risk='TRACEBACK'
elif status.get('pipeline_status')=='FAILED':risk='PIPELINE_FAILED'
derived='COMPLETE' if receipt.exists() else ('RUNNING' if wp or cp else status.get('pipeline_status','NOT_RUNNING'))
vals={
'TIMESTAMP':datetime.now().astimezone().isoformat(timespec='seconds'),'RUN_ROOT':RUN,'PIPELINE_STATUS_DERIVED':derived,
**{f'{c}_COMPLETED':counts[c] for c in ['F00','F10','F01','F11']},'TOTAL_BRANCHES_COMPLETED':len(valid),'TOTAL_BRANCHES_EXPECTED':668,'FOUR_CELL_SAMPLES_COMPLETED':four,'FOUR_CELL_SAMPLES_EXPECTED':167,'PERCENT_COMPLETE':f'{len(valid)/668*100:.3f}',
'TMUX_SESSION':SESSION,'TMUX_EXISTS':'YES' if tmux_ok else 'NO','CONTROLLER_PID':','.join(map(str,cp)) or 'NONE','CONTROLLER_RUNNING':'YES' if cp else 'NO','WORKER_PID':worker or 'NONE','WORKER_RUNNING':'YES' if worker else 'NO','WORKER_CPU_AFFINITY':aff or 'NONE',
'CURRENT_SAMPLE':status.get('current_sample','NONE'),'CURRENT_TASK':status.get('current_task','NONE'),'CURRENT_CELL':status.get('current_cell','NONE'),'CURRENT_BOUNDARY':status.get('current_boundary','NONE'),'TEMP_RESULTS':len(temps),'ERROR_COUNT':len(errs),'DUPLICATE_BRANCH_IDS':dups,
'LAST_COMMIT_TIMESTAMP':datetime.fromtimestamp(last).astimezone().isoformat(timespec='seconds') if last else 'NONE','SECONDS_SINCE_LAST_COMMIT':int(time.time()-last) if last else 'UNKNOWN','RECENT_BRANCHES_PER_HOUR':f'{rate:.3f}','ETA':eta,
'GPU_MODEL':gpu[0],'GPU_UTILIZATION':gpu[1],'GPU_MEMORY_USED':gpu[2],'GPU_MEMORY_TOTAL':gpu[3],'GPU_MEMORY_FREE':gpu[4],'GPU_TEMPERATURE':gpu[5],'GPU_POWER':gpu[6],
'OOM_DETECTED':'YES' if 'out of memory' in log.lower() else 'NO','TRACEBACK_DETECTED':'YES' if 'Traceback' in log else 'NO','STALL_WARNING':'YES' if last and time.time()-last>3600 and wp else 'NO','FORMAL_RESULT_PIPELINE_RISK':risk,'F11_CLOSURE_FAILURES_SO_FAR':sum(x.get('cell')=='F11' and not x.get('closure',{}).get('L6') for x in valid),'FORMAL_COMPLETE':'YES' if len(valid)==668 else 'NO','FINAL_RECEIPT_EXISTS':'YES' if receipt.exists() else 'NO'}
for k,v in vals.items():print(f'{k}={v}')
