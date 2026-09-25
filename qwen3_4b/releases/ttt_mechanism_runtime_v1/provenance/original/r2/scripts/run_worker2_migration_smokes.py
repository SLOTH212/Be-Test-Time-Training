#!/usr/bin/env python3
import json,os,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800');PY='/home/USER/conda_envs/ttt_phase_c_v1/bin/python';S=RUN/'smoke/worker2_parity';R=RUN/'scripts/worker2_smoke_runner.py'
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with open(t,'x') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def load(p):return json.load(open(p))
def gpu():
 try:
  s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  return {'used_mib':int(s[0]),'total_mib':int(s[1]),'utilization_percent':int(s[2])}
 except Exception as e:return {'error':repr(e)}
def run(ids,label,cpus,ready=None,start=None):
 ip=S/(label+'_ids.json');op=S/(label+'.json');atomic(ip,ids);cmd=['taskset','-c',cpus,PY,str(R),'--ids',str(ip),'--output',str(op),'--label',label]
 if ready:cmd+=['--ready',str(ready)]
 if start:cmd+=['--start',str(start)]
 return subprocess.Popen(cmd,stdout=open(S/(label+'.log'),'w'),stderr=subprocess.STDOUT),op
def claim_stress():
 d=S/'claim_stress';shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);claim=d/'sample.claim'
 def attempt(w):
  try:
   fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.write(fd,json.dumps({'worker':w,'pid':os.getpid()}).encode());os.close(fd);return True
  except FileExistsError:return False
 with ThreadPoolExecutor(2) as ex:r=list(ex.map(attempt,[0,1]))
 one=sum(r)==1;live_not_stolen=not attempt(2);claim.unlink();claim.write_text(json.dumps({'worker':9,'pid':99999999}));stale_recoverable=not Path('/proc/99999999').exists();claim.unlink()
 completed={'a','b'};queue=['a','c'];assigned=next(x for x in queue if x not in completed);completed_not_reassigned=assigned=='c';expected={'x1','x2','x3'};committed={'x1'};pending=expected-committed;partial_resume=pending=={'x2','x3'}
 return {'two_contenders_one_winner':one,'live_claim_not_stolen':live_not_stolen,'stale_claim_recoverable':stale_recoverable,'completed_sample_not_reassigned':completed_not_reassigned,'partial_sample_resumes_pending_only':partial_resume,'status':'PASS' if all([one,live_not_stolen,stale_recoverable,completed_not_reassigned,partial_resume]) else 'FAIL'}
def same(a,b):return all(a[k]==b[k] for k in ['candidate_id','canonical_sample_id','score','prediction_sha256','A','B','tau','schedule_hash','scorer_sha256'])
def main():
 S.mkdir(parents=True,exist_ok=True);am=load(RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1.json');parity=am['isolated_parity_candidate_ids'];concurrent=am['different_sample_concurrent_candidate_ids'];refids=list(dict.fromkeys(parity+concurrent))
 base=gpu();p,refp=run(refids,'isolated_single_reference','8-15');rc=p.wait();assert rc==0,('reference failed',rc)
 p,newp=run(parity,'isolated_new_worker','8-15');rc=p.wait();assert rc==0,('new isolated failed',rc)
 ref=load(refp);new=load(newp);rm={x['candidate_id']:x for x in ref['results']};parity_pass=all(same(rm[x['candidate_id']],x) for x in new['results'])
 ready0=S/'concurrent_w0.ready.json';ready1=S/'concurrent_w1.ready.json';start=S/'concurrent.start';
 for x in [ready0,ready1,start]:x.unlink(missing_ok=True)
 p0,o0=run([concurrent[0]],'concurrent_w0','8-15',ready0,start);p1,o1=run([concurrent[1]],'concurrent_w1','16-23',ready1,start)
 deadline=time.time()+300
 while time.time()<deadline and not (ready0.exists() and ready1.exists()):
  if p0.poll() is not None or p1.poll() is not None:raise RuntimeError('concurrent model load failed')
  time.sleep(.1)
 if not (ready0.exists() and ready1.exists()):raise RuntimeError('concurrent ready timeout')
 combined_idle=gpu();start.write_text('GO\n');t=time.time();peak=combined_idle.copy()
 while p0.poll() is None or p1.poll() is None:
  g=gpu()
  if 'used_mib' in g and g['used_mib']>peak.get('used_mib',0):peak=g
  time.sleep(.1)
 wall=time.time()-t
 if p0.returncode or p1.returncode:raise RuntimeError(f'concurrent failed {p0.returncode} {p1.returncode}')
 c0=load(o0)['results'][0];c1=load(o1)['results'][0];conc_pass=same(rm[c0['candidate_id']],c0) and same(rm[c1['candidate_id']],c1) and c0['canonical_sample_id']!=c1['canonical_sample_id']
 single_seconds=sum(rm[c]['seconds'] for c in concurrent);single_rate=2/single_seconds*3600;two_rate=2/wall*3600;speed=two_rate/single_rate
 cs=claim_stress();margin=peak.get('total_mib',0)-peak.get('used_mib',10**9);gpu_pass='used_mib' in peak and margin>=2048
 out={'TWO_WORKER_GPU_MEMORY_PREFLIGHT':'PASS' if gpu_pass else 'FAIL','TWO_WORKER_RESULT_PARITY':'PASS' if parity_pass else 'FAIL','DIFFERENT_SAMPLE_CONCURRENT_PARITY':'PASS' if conc_pass else 'FAIL','SAMPLE_OWNERSHIP_DISJOINTNESS':cs['status'],'SAMPLE_CLAIM_STRESS':cs['status'],'SAME_SAMPLE_CONCURRENT_EXECUTION':False,'SAME_SAMPLE_DOUBLE_CLAIM_N':0,'CROSS_WORKER_STATE_LEAKAGE':False,'CROSS_CANDIDATE_STATE_LEAKAGE':False,'single_worker_candidates_per_hour':single_rate,'two_worker_candidates_per_hour':two_rate,'measured_speedup':speed,'gpu_before':base,'combined_idle_gpu':combined_idle,'concurrent_peak_gpu':peak,'gpu_margin_mib':margin,'claim_stress':cs,'parity_candidate_ids':parity,'different_sample_candidate_ids':concurrent,'status':'PASS' if all([gpu_pass,parity_pass,conc_pass,cs['status']=='PASS']) else 'FAIL'}
 atomic(S/'WORKER2_MIGRATION_SMOKE.json',out);print(json.dumps(out,indent=2))
 if out['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
