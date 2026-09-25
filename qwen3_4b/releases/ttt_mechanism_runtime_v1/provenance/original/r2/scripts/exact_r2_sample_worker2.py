#!/usr/bin/env python3
"""Two-process Exact-R2 worker with exclusive sample ownership."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, math, os, time, traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
MODEL_SHA='ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'; CKPT_SHA='f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
BENCH_SHA='5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5'; RULE_SHA='afeb7f0ceb42cfaa753ea04d926866ba346f83015be92ff8ad5138d5c32f8378'
ELIG_SHA='f966cbfd65f8f9fbc0d079e44cd5e46fb24ca7da72130f5200c84be6faee3a42'; CAND_SHA='737a5244a561e103c660ae33aff7a94e9c53e195d1fc05fc1006a1a7b4979739'
ORIGINAL_PROTOCOL_SHA='b3cd7b596169499d92816ab043206a9dba32ebc0a74ece7225343213ec09fd76'

def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):
 with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def alive(pid):
 try:os.kill(int(pid),0);return True
 except Exception:return False

old=imp(RUN/'scripts/exact_r2_worker.py','exact_r2_original_worker_authority')

def gate(worker_id):
 cfg=load(RUN/'config/EXACT_R2_FORMAL_CONFIG.json');mcfg=load(RUN/'config/EXECUTION_SAMPLE_PARALLEL_WORKERS2_V1_CONFIG.json')
 expected_affinity=list(range(8,16)) if worker_id==0 else list(range(16,24))
 checks={'model':sha(Path(cfg['model_root'])/'model.safetensors')==MODEL_SHA,'benchmark':sha(Path(cfg['benchmark_root'])/'benchmark_manifest.json')==BENCH_SHA,
  'rule':sha(cfg['rule_manifest'])==RULE_SHA,'eligibility':sha(cfg['eligibility_manifest'])==ELIG_SHA,
  'candidate':sha(cfg['candidate_manifest'])==CAND_SHA,'protocol':sha(cfg['protocol'])==ORIGINAL_PROTOCOL_SHA,
  'amendment':sha(mcfg['execution_amendment'])==mcfg['execution_amendment_sha256'],
  'worker2':sha(mcfg['worker2'])==mcfg['worker2_sha256'],'controller2':sha(mcfg['controller2'])==mcfg['controller2_sha256'],
  'affinity':sorted(os.sched_getaffinity(0))==expected_affinity}
 if not all(checks.values()):raise RuntimeError('WORKER2_GATE '+json.dumps(checks,sort_keys=True))
 manifest=rows(cfg['candidate_manifest']);by=defaultdict(list)
 for c in manifest:by[c['canonical_sample_id']].append(c)
 if len(manifest)!=99120 or len(by)!=173:raise RuntimeError('MANIFEST_SCOPE')
 return cfg,mcfg,manifest,by,checks

def claim_path(sid):return RUN/'sample_claims'/f'{old.sample_key(sid)}.claim'
def status_path(worker_id):return RUN/'status'/f'worker{worker_id}.json'
def status(worker_id,**kw):
 p=status_path(worker_id);x=load(p) if p.exists() else {};x.update(kw);x['updated_at']=now();old.atomic(p,x)

def stale_safe(sid,claim):
 try:x=load(claim)
 except Exception:return False
 if alive(x.get('worker_pid')):return False
 if any((RUN/'branches'/old.sample_key(sid)).glob('.*.tmp')):return False
 stale=RUN/'sample_claims/stale'/f'{claim.name}.{time.time_ns()}.stale'
 stale.parent.mkdir(parents=True,exist_ok=True)
 try:os.replace(claim,stale)
 except FileNotFoundError:return False
 d=os.open(stale.parent,os.O_RDONLY);os.fsync(d);os.close(d)
 old.atomic(RUN/'audit'/f'STALE_SAMPLE_CLAIM_RECOVERY_{time.time_ns()}.json',{'sample_id':sid,'old_claim':x,'stale_path':str(stale),'recovered_at':now()})
 return True

def acquire(sid,worker_id,mcfg):
 p=claim_path(sid);p.parent.mkdir(parents=True,exist_ok=True)
 for _ in range(2):
  payload={'sample_id':sid,'worker_id':worker_id,'worker_pid':os.getpid(),'controller_pid':int(os.environ['EXACT_R2_WORKER2_CONTROLLER_PID']),
           'claimed_at':now(),'execution_amendment_sha256':mcfg['execution_amendment_sha256'],'parallelization_unit':'SAMPLE'}
  try:
   fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
   with os.fdopen(fd,'w') as f:json.dump(payload,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
   d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d);return p,payload
  except FileExistsError:
   if not stale_safe(sid,p):return None,None
 return None,None

def release(p):
 os.unlink(p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)

def runtime(worker_id,cfg):
 import torch
 torch.set_num_threads(8);runner=old.imp(cfg['frozen_executor'],f'exact_r2_worker2_executor_{worker_id}');ctx=old.Context(cfg)
 status(worker_id,state='MODEL_LOADING',pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
 model,tok=runner.load_model(ctx);model.requires_grad_(False);base=runner.base_hash(model)
 old.atomic(RUN/'audit'/f'MODEL_LOAD_WORKER{worker_id}.json',{'status':'PASS','worker_id':worker_id,'pid':os.getpid(),'model_identity':MODEL_SHA,
  'checkpoint_identity':CKPT_SHA,'base_weight_runtime_hash':base,'cuda_device':torch.cuda.get_device_name(0),
  'cuda_memory_allocated_bytes':torch.cuda.memory_allocated(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'timestamp':now()})
 return runner,ctx,model,tok,base

def make_result(c,rec,pred,cfg,mcfg,worker_id,started,wall):
 return {'candidate_id':c['candidate_id'],'canonical_sample_id':c['canonical_sample_id'],'task':c['task'],'benchmark_global_index':c['benchmark_global_index'],
  'A':c['action_A'],'B':c['action_B'],'tau':c['tau'],'T':c['T'],'regime_count':2,'switch_count':1,
  'full_schedule_hash':c['canonical_schedule_hash'],'score':float(rec['score']),'prediction':pred,'prediction_sha256':rec['prediction_hash'],
  'execution_status':'SUCCESS','SUCCESS':True,'model_identity':MODEL_SHA,'checkpoint_identity':CKPT_SHA,'benchmark_identity':BENCH_SHA,
  'rule_sha256':RULE_SHA,'eligibility_sha256':ELIG_SHA,'candidate_manifest_sha256':CAND_SHA,'protocol_sha256':ORIGINAL_PROTOCOL_SHA,
  'executor_sha256':old.EXECUTOR_SHA,'scorer_sha256':old.SCORER_SHA,'clean_full_replay':True,'normal_per_sample_initial_state':True,
  'candidate_prefix_reuse':False,'dynamic_state_transplant':False,'cross_candidate_ttt_state_reuse':False,'stored_delta_stitching':False,
  'recompute_downstream_updates':True,'generation_update_count':rec['generation_update_count'],'layer_update_counts':rec['layer_update_counts'],
  'prompt_token_count':rec['prompt_token_count'],'incomplete_tail_length':rec['prompt_token_count']%1024,
  'gpu_peak_memory_bytes':rec['gpu_peak_memory_bytes'],'runtime_seconds':rec['latency_seconds'],'wall_seconds':wall,
  'started_at':started,'completed_at':now(),'worker_id':worker_id,'worker_count':2,
  'execution_amendment_sha256':mcfg['execution_amendment_sha256'],'sample_level_exclusive_ownership':True}

def finalize_sample(sid,candidates,records,cfg):
 win,best,ties=old.choose_winner(records)
 out={'canonical_sample_id':sid,'task':candidates[0]['task'],'winning_candidate_id':win['candidate_id'],'R2_score':best,
  'winning_A':win['A'],'winning_B':win['B'],'winning_tau':win['tau'],'T':win['T'],'switch_count':1,'regime_count':2,
  'number_of_max_score_ties':ties,'tie_break_trace':['maximum score','earliest tau','action order A','action order B','sequence SHA256'],
  'score_tolerance':old.TOL,'candidate_count':len(candidates),'candidate_manifest_sha256':CAND_SHA,
  'protocol_sha256':ORIGINAL_PROTOCOL_SHA,'completed_at':now()}
 p=old.winner_path(RUN,sid)
 if p.exists():
  prior=load(p)
  if prior['winning_candidate_id']!=out['winning_candidate_id'] or abs(prior['R2_score']-out['R2_score'])>old.TOL:raise RuntimeError('WINNER_CONFLICT '+sid)
 else:old.atomic(p,out)

def main(worker_id):
 import torch
 cfg,mcfg,manifest,by,checks=gate(worker_id);status(worker_id,state='STARTING',pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
  owned_sample=None,current_candidate=None,worker_running=True,last_error='NONE')
 completed_samples=set();runner,ctx,model,tok,base=runtime(worker_id,cfg);new=0;started=time.time()
 try:
  while True:
   owned=None
   for sid,candidates in by.items():
    if sid in completed_samples:continue
    complete=True
    for c in candidates:
     existing=old.branch_path(RUN,c)
     if not existing.exists():complete=False;break
     old.valid_result(existing,c,cfg)
    if complete and old.winner_path(RUN,sid).exists():completed_samples.add(sid);continue
    p,claim=acquire(sid,worker_id,mcfg)
    if p is not None:owned=(sid,candidates,p);break
   if owned is None:break
   sid,candidates,cp=owned;status(worker_id,state='RUNNING',owned_sample=sid,owned_task=candidates[0]['task'],sample_candidates_expected=len(candidates))
   records=[]
   for c in candidates:
    path=old.branch_path(RUN,c)
    if path.exists():records.append(old.valid_result(path,c,cfg));continue
    status(worker_id,current_candidate=c['candidate_id'],current_A=c['action_A'],current_B=c['action_B'],current_tau=c['tau'],sample_candidates_completed=len(records))
    st=now();wall=time.time();rec,pred=old.execute(runner,ctx,model,tok,ctx.by_id[sid],c['full_chunk_level_action_sequence'],
      f'formal_exact_r2_worker2_sample_parallel_w{worker_id}',1,1,RUN/'status'/f'worker{worker_id}_internal_progress.json')
    if runner.base_hash(model)!=base:raise RuntimeError('BASE_WEIGHT_MUTATION '+c['candidate_id'])
    out=make_result(c,rec,pred,cfg,mcfg,worker_id,st,time.time()-wall);old.atomic(path,out);old.valid_result(path,c,cfg);records.append(out);new+=1
    rate=new/max(time.time()-started,1e-9)*3600;status(worker_id,current_candidate=c['candidate_id'],sample_candidates_completed=len(records),
      last_commit_timestamp=out['completed_at'],new_candidates_committed=new,recent_candidates_per_hour=rate)
    if new%10==0:print(json.dumps({'worker':worker_id,'sample':sid,'sample_progress':f'{len(records)}/{len(candidates)}','candidate':c['candidate_id']},sort_keys=True),flush=True)
    del rec,pred
    if new%25==0:gc.collect();torch.cuda.empty_cache()
   if len(records)!=len(candidates):raise RuntimeError('SAMPLE_INCOMPLETE '+sid)
   finalize_sample(sid,candidates,records,cfg);release(cp);completed_samples.add(sid);status(worker_id,state='SAMPLE_FINALIZED',owned_sample=None,current_candidate=None,sample_candidates_completed=len(records))
   del records;gc.collect();torch.cuda.empty_cache()
  status(worker_id,state='COMPLETE',owned_sample=None,current_candidate=None,worker_running=False,new_candidates_committed=new,completed_at=now())
 except BaseException as e:
  old.atomic(RUN/'errors'/f'worker2_{worker_id}_{time.time_ns()}.json',{'worker_id':worker_id,'failure_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'timestamp':now()})
  status(worker_id,state='FAILED',worker_running=False,last_error=f'{type(e).__name__}: {e}');raise
 finally:
  del model,tok;gc.collect();import torch;torch.cuda.empty_cache()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker-id',type=int,choices=[0,1],required=True);a=p.parse_args();main(a.worker_id)
