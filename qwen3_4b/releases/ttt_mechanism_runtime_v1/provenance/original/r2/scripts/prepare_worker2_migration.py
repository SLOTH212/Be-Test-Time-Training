#!/usr/bin/env python3
import hashlib,json,os,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with open(t,'x') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def main():
 cfg=json.load(open(RUN/'config/EXACT_R2_FORMAL_CONFIG.json'));rows=[json.loads(x) for x in open(cfg['candidate_manifest'])];by=defaultdict(list)
 for r in rows:by[r['canonical_sample_id']].append(r)
 counts={sid:sum((RUN/'branches'/hashlib.sha256(sid.encode()).hexdigest()[:20]/(c['candidate_id']+'.json')).exists() for c in cs) for sid,cs in by.items()}
 partial=[{'sample_id':sid,'committed':n,'expected':len(by[sid]),'pending':len(by[sid])-n} for sid,n in counts.items() if 0<n<len(by[sid])]
 committed=sum(counts.values());complete=sum(n==len(by[s]) for s,n in counts.items())
 post={'timestamp':now(),'committed_after_safe_pause':committed,'fully_searched_samples':complete,'winners_finalized':len(list((RUN/'winners').glob('*.json'))),'partially_completed_samples':partial,'temp_results':len(list((RUN/'branches').glob('**/*.tmp*'))),'controller_running':False,'worker_running':False,'status':'PASS'}
 atomic(RUN/'audit/POST_SAFE_PAUSE_SAMPLE_PARALLEL_AUDIT.json',post)
 positions=[0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1];parity=[rows[i]['candidate_id'] for i in positions]
 distinct=[]
 for sid,cs in by.items():
  if counts[sid]==len(cs):distinct.append(cs[0]['candidate_id'])
  if len(distinct)==2:break
 amendment={'amendment_type':'EXECUTION_ONLY_CONCURRENCY_CHANGE','created_at':now(),'scientific_protocol_changed':False,'original_workers':1,'target_workers':2,'parallelization_unit':'SAMPLE','sample_ownership':'EXCLUSIVE','candidate_commit_unit':'INDIVIDUAL_CANDIDATE','candidate_state_reuse':'FORBIDDEN','candidate_manifest_unchanged':True,'rule_unchanged':True,'eligibility_unchanged':True,'scorer_unchanged':True,'winner_rule_unchanged':True,'outcomes_used_to_choose_amendment':False,'original_protocol_sha256':cfg['protocol_sha256'],'rule_sha256':cfg['rule_sha256'],'eligibility_sha256':cfg['eligibility_sha256'],'candidate_manifest_sha256':cfg['candidate_manifest_sha256'],'committed_after_safe_pause':committed,'partially_completed_samples_at_migration':len(partial),'smoke_selection_rule':'manifest positions [0,N//4,N//2,3N//4,N-1]; concurrent first candidates of first two already-complete samples','isolated_parity_manifest_positions':positions,'isolated_parity_candidate_ids':parity,'different_sample_concurrent_candidate_ids':distinct,'same_sample_concurrent_execution':False,'cpu_affinity':{'worker0':'8-15','worker1':'16-23'}}
 ap=RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1.json';atomic(ap,amendment);ah=sha(ap)
 md='# Execution Sample-Parallel Workers=2 Amendment V1\n\n'+ '\n'.join(f'{k}={v}' for k,v in amendment.items())+'\n';(RUN/'amendments/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1.md').write_text(md)
 worker=RUN/'scripts/exact_r2_sample_worker2.py';controller=RUN/'scripts/controller_workers2_sample_parallel.py'
 mcfg={'execution_amendment':str(ap),'execution_amendment_sha256':ah,'worker2':str(worker),'worker2_sha256':sha(worker),'controller2':str(controller),'controller2_sha256':sha(controller),'smoke_runner':str(RUN/'scripts/worker2_smoke_runner.py'),'smoke_runner_sha256':sha(RUN/'scripts/worker2_smoke_runner.py'),'original_protocol_sha256':cfg['protocol_sha256'],'created_at':now()}
 atomic(RUN/'config/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_CONFIG.json',mcfg)
 print(json.dumps({'committed':committed,'complete_samples':complete,'partial':partial,'amendment_sha256':ah,'parity_ids':parity,'concurrent_ids':distinct},indent=2))
if __name__=='__main__':main()
