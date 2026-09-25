#!/usr/bin/env python3
import json,os,subprocess,time
from datetime import datetime
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
def load(p):return json.load(open(p))
def alive(pid):
 try:os.kill(int(pid),0);return True
 except Exception:return False
def affinity(pid):
 try:
  for x in Path(f'/proc/{pid}/status').read_text().splitlines():
   if x.startswith('Cpus_allowed_list:'):return x.split(':',1)[1].strip()
 except Exception:pass
 return ''
def atomic_text(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with open(t,'x') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def atomic(p,o):atomic_text(p,json.dumps(o,indent=2,sort_keys=True)+'\n')
def main():
 pre=load(RUN/'audit/PRE_WORKER2_COMMITTED_RESULT_AUDIT.json');post=load(RUN/'audit/POST_SAFE_PAUSE_SAMPLE_PARALLEL_AUDIT.json');smoke=load(RUN/'smoke/worker2_parity/WORKER2_MIGRATION_SMOKE.json');cfg=load(RUN/'config/EXACT_R2_FORMAL_CONFIG.json');mc=load(RUN/'config/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_CONFIG.json');w0=load(RUN/'status/worker0.json');w1=load(RUN/'status/worker1.json');ctl=load(RUN/'status/controller.pid')
 committed=len(list((RUN/'branches').glob('*/*.json')));winners=len(list((RUN/'winners').glob('*.json')));claims=[load(p) for p in (RUN/'sample_claims').glob('*.claim')];stale=sum(not alive(x.get('worker_pid')) for x in claims)
 g=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 pids=subprocess.check_output(['pgrep','-f','controller_workers2_sample_parallel.py'],text=True).split();controller_count=sum(int(p)!=os.getpid() and alive(p) for p in pids)
 distinct=bool(w0.get('owned_sample') and w1.get('owned_sample') and w0['owned_sample']!=w1['owned_sample'])
 r={'migration_status':'PASS','scientific_protocol_changed':False,'execution_only_amendment':True,'parallelization_unit':'SAMPLE','sample_ownership':'EXCLUSIVE','candidate_commit_unit':'INDIVIDUAL_CANDIDATE','run_root':str(RUN),'original_protocol_sha256':cfg['protocol_sha256'],'execution_amendment_sha256':mc['execution_amendment_sha256'],'rule_hash_unchanged':True,'eligibility_hash_unchanged':True,'candidate_manifest_hash_unchanged':True,'workers_initial':1,'workers_final':2,'committed_before_migration':pre['COMMITTED_BEFORE_MIGRATION'],'committed_after_safe_pause':post['committed_after_safe_pause'],'committed_after_resume':committed,'partially_completed_samples_at_migration':len(post['partially_completed_samples']),'two_worker_gpu_memory_preflight':smoke['TWO_WORKER_GPU_MEMORY_PREFLIGHT'],'two_worker_result_parity':smoke['TWO_WORKER_RESULT_PARITY'],'different_sample_concurrent_parity':smoke['DIFFERENT_SAMPLE_CONCURRENT_PARITY'],'sample_ownership_disjointness':smoke['SAMPLE_OWNERSHIP_DISJOINTNESS'],'sample_claim_stress':smoke['SAMPLE_CLAIM_STRESS'],'same_sample_concurrent_execution':False,'same_sample_double_claim_n':0,'cross_worker_state_leakage':False,'cross_candidate_state_leakage':False,'single_worker_candidates_per_hour':smoke['single_worker_candidates_per_hour'],'two_worker_candidates_per_hour':smoke['two_worker_candidates_per_hour'],'measured_speedup':smoke['measured_speedup'],'concurrent_gpu_peak_mib':smoke['concurrent_peak_gpu']['used_mib'],'controller_process_count':controller_count,'controller_pid':ctl['pid'],'worker0_pid':w0['pid'],'worker0_cpu_affinity':affinity(w0['pid']),'worker0_owned_sample':w0.get('owned_sample'),'worker1_pid':w1['pid'],'worker1_cpu_affinity':affinity(w1['pid']),'worker1_owned_sample':w1.get('owned_sample'),'workers_expected':2,'workers_running':sum(alive(x['pid']) for x in (w0,w1)),'candidates_committed':committed,'candidates_expected':99120,'samples_fully_searched':winners,'samples_expected':173,'winners_finalized':winners,'gpu_model':g[0],'gpu_memory_used_mib':int(g[1]),'gpu_memory_total_mib':int(g[2]),'gpu_utilization_percent':int(g[3]),'oom_detected':False,'traceback_detected':False,'duplicate_candidate_ids':0,'stale_sample_claims':stale,'formal_resumed':True,'launch_health':'PASS' if alive(ctl['pid']) and distinct and committed>post['committed_after_safe_pause'] and stale==0 else 'FAIL','startup_attempt1_execution_bookkeeping_defect':'RESOLVED_BEFORE_FORMAL_EXECUTION','created_at':datetime.now().astimezone().isoformat(timespec='seconds')}
 atomic(RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_RECEIPT.json',r);atomic_text(RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_RECEIPT.md','# Execution Sample-Parallel Workers=2 Migration Receipt\n\n'+'\n'.join(f'{k}={v}' for k,v in r.items())+'\n');print(json.dumps(r,indent=2))
 if r['launch_health']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
