"""Stop only identity-verified predecessor processes after both GPU smokes pass."""
import os,signal,time,subprocess
from pathlib import Path
import run as r

def identity(pid):
 p=Path('/proc')/str(pid)
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),start_ticks=fields[19],uid=p.stat().st_uid)

def alive(pid):
 p=Path('/proc')/str(pid)
 try:return (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def main():
 j=r.verify(full=True)
 assert not (r.O/'STOP_RECEIPT.json').exists()
 smoke=r.read(r.O/'SMOKE_STATUS.json');assert smoke['status']=='PASS' and smoke['job_sha256']==r.sha(r.S/'JOB.json')
 smoke_pid=r.read(r.O/'SMOKE_LAUNCH.json')['pid'];assert not alive(smoke_pid),'SMOKE_MANAGER_STILL_HOLDS_GPU5_LOCK'
 for label in j['models']:
  s=r.read(r.O/label/'SMOKE.json');assert s['status']=='PASS' and s['job_sha256']==r.sha(r.S/'JOB.json')
 old=r.read(r.OLD_O/'PHASE_STATUS.json');assert old['status']=='RUNNING' and old['job_sha256']==j['predecessor_job_sha256']
 manager=old['manager_pid'];workers=[]
 for label in j['models']:
  p=r.OLD_O/label/'PHASE_STATUS.json'
  if p.exists():
   d=r.read(p)
   if d['status']=='RUNNING':workers+=d['pids']
 assert len(workers)==7
 ids=[identity(pid) for pid in [manager]+workers]
 for i in ids:
  assert i['uid']==os.getuid() and str(r.OLD_S/'run.py') in i['cmdline']
 assert ' manager' in ids[0]['cmdline'] and all(' worker ' in i['cmdline'] for i in ids[1:])
 children=subprocess.check_output(['ps','--ppid',str(manager),'-o','pid='],text=True)
 assert {int(v) for v in children.split()}==set(workers)
 r.atomic(r.O/'STOP_INTENT.json',dict(identities=ids,time=time.time(),reason='User-authorized eight GPU migration; new KV and historical Selective; preserve committed results'))
 assert identity(manager)==ids[0]
 os.kill(manager,signal.SIGSTOP)
 try:
  deadline=time.time()+5
  while (Path('/proc')/str(manager)/'stat').read_text().rsplit(')',1)[1].split()[0] not in ['T','t']:
   assert time.time()<deadline,'OLD_MANAGER_NOT_STOPPED'
   time.sleep(.05)
  for i in ids[1:]:
   assert identity(i['pid'])==i
   os.kill(i['pid'],signal.SIGTERM)
  deadline=time.time()+35
  while any(alive(pid) for pid in workers) and time.time()<deadline:time.sleep(.25)
  assert not any(alive(pid) for pid in workers),'OLD_WORKER_DID_NOT_EXIT'
  assert identity(manager)==ids[0]
  os.kill(manager,signal.SIGTERM);os.kill(manager,signal.SIGCONT)
  deadline=time.time()+15
  while any((Path('/proc')/str(pid)).exists() for pid in [manager]+workers) and time.time()<deadline:time.sleep(.25)
  assert not any((Path('/proc')/str(pid)).exists() for pid in [manager]+workers),'OLD_PROCESS_NOT_REAPED'
 except BaseException:
  if alive(manager):os.kill(manager,signal.SIGCONT)
  raise
 r.atomic(r.O/'STOP_RECEIPT.json',dict(status='PASS',all_old_processes_gone=True,old_pids=[manager]+workers,identities=ids,time=time.time()))
 print('OLD_PROCESSES_STOPPED_AND_REAPED',flush=True)
 r.import_snapshot()
 print('IMPORT_PASS',r.read(r.O/'IMPORT_MANIFEST.json')['summary'],flush=True)
 r.atomic(r.OLD_O/'SUPERSEDED_BY.json',dict(status='SUPERSEDED',successor_output=str(r.O),successor_job_sha256=r.sha(r.S/'JOB.json'),stop_receipt=str(r.O/'STOP_RECEIPT.json'),import_manifest=str(r.O/'IMPORT_MANIFEST.json'),time=time.time()))
 # New manager holds every GPU lock for its entire lifetime and refuses weak headroom.
 log=(r.O/'formal_manager.log').open('a')
 p=subprocess.Popen([os.sys.executable,'-B',str(r.S/'run.py'),'manager'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r.S)
 r.atomic(r.O/'LAUNCH.json',dict(pid=p.pid,source=str(r.S),output=str(r.O),job_sha256=r.sha(r.S/'JOB.json'),gpus=r.GPUS,time=time.time()))
 print('NEW_MANAGER',p.pid,flush=True)

if __name__=='__main__':main()
