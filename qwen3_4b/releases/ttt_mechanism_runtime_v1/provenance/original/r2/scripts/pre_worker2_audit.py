#!/usr/bin/env python3
"""Capture execution metadata and validate immutable commits before worker migration."""
import hashlib, importlib.util, json, os, subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
WORKER=RUN/'scripts/exact_r2_worker.py'; TMUX='/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/tools/tmux'
def imp(p):
 s=importlib.util.spec_from_file_location('r2w_pre_migration',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def alive(pid):
 try:os.kill(int(pid),0);return True
 except Exception:return False
def gpu():
 x=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout.strip().split(', ')
 return {'model':x[0],'utilization_percent':float(x[1]),'memory_used_mib':int(x[2]),'memory_total_mib':int(x[3]),'memory_free_mib':int(x[4])}
w=imp(WORKER);run,cfg,manifest,by_sample,mappings,checks=w.gates(RUN);status=w.load(RUN/'status/pipeline_status.json')
valid=[];invalid=[];latest=None
for c in manifest:
 p=w.branch_path(RUN,c)
 if not p.exists():continue
 try:
  w.valid_result(p,c,cfg);valid.append(c['candidate_id']);mt=p.stat().st_mtime
  if latest is None or mt>latest[0]:latest=(mt,c['candidate_id'],str(p))
 except Exception as e:invalid.append({'candidate_id':c['candidate_id'],'error':str(e)})
counts=Counter(c['canonical_sample_id'] for c in manifest);done=Counter();valid_set=set(valid)
for c in manifest:
 if c['candidate_id'] in valid_set:done[c['canonical_sample_id']]+=1
controller=status.get('controller_pid');worker=status.get('worker_pid');tmux=subprocess.run([TMUX,'has-session','-t','exact_r2_1p7b_v1'],capture_output=True).returncode==0
state={'timestamp':datetime.now().astimezone().isoformat(timespec='seconds'),'run_root':str(RUN),'tmux_session':'exact_r2_1p7b_v1','tmux_exists':tmux,
 'controller_pid':controller,'controller_running':alive(controller),'worker_pid':worker,'worker_running':alive(worker),
 'worker_cpu_affinity':sorted(os.sched_getaffinity(int(worker))) if alive(worker) else [],'committed_candidates':len(valid),
 'fully_searched_samples':sum(done[s]==counts[s] for s in counts),'finalized_winners':len(list((RUN/'winners').glob('*.json'))),
 'current_sample':status.get('current_sample'),'current_candidate':status.get('current_candidate'),
 'temp_partial_candidates':sum(1 for _ in RUN.glob('branches/*/.*.tmp')),
 'error_count':len(list((RUN/'errors').glob('*.json'))),'last_committed_candidate':latest[1] if latest else None,
 'last_commit_timestamp':datetime.fromtimestamp(latest[0]).astimezone().isoformat(timespec='seconds') if latest else None,
 'last_commit_path':latest[2] if latest else None,'gpu':gpu(),'formal_complete':(RUN/'reports/EXACT_R2_1P7B_FORMAL_COMPLETION_RECEIPT.json').exists()}
audit={'status':'PASS' if not invalid and len(valid)==len(set(valid)) else 'FAIL','COMMITTED_BEFORE_MIGRATION':len(valid),
 'unique_candidate_ids':len(set(valid)),'valid_success_results':len(valid),'invalid_results':len(invalid),'invalid_examples':invalid[:20],
 'model_identity':w.MODEL_SHA,'checkpoint_identity':w.CKPT_SHA,'rule_sha256':w.RULE_SHA,'eligibility_sha256':w.ELIG_SHA,
 'candidate_manifest_sha256':w.CAND_SHA,'original_protocol_sha256':cfg['protocol_sha256'],'state':state}
w.atomic(RUN/'audit/PRE_WORKER2_SAMPLE_PARALLEL_MIGRATION_STATE.json',state)
w.atomic(RUN/'audit/PRE_WORKER2_COMMITTED_RESULT_AUDIT.json',audit)
md='# Pre Worker-2 Migration State\n\n'+ '\n'.join(f'{k}={v}' for k,v in state.items() if not isinstance(v,dict))+'\n\nGPU='+json.dumps(state['gpu'],sort_keys=True)+'\n'
w.atomic_text(RUN/'audit/PRE_WORKER2_SAMPLE_PARALLEL_MIGRATION_STATE.md',md)
print(json.dumps({'state':state,'audit_status':audit['status']},indent=2,sort_keys=True))
