#!/usr/bin/env python3
"""Smoke-only clean replay process; never writes formal branches or winners."""
import argparse,gc,hashlib,importlib.util,json,os,time
from pathlib import Path
RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
def imp(p,n):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 with open(t,'x') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ids',required=True);ap.add_argument('--output',required=True);ap.add_argument('--ready');ap.add_argument('--start');ap.add_argument('--label',required=True);a=ap.parse_args()
 import torch
 torch.set_num_threads(len(os.sched_getaffinity(0)));old=imp(RUN/'scripts/exact_r2_worker.py','smoke_authority_'+str(os.getpid()));cfg=json.load(open(RUN/'config/EXACT_R2_FORMAL_CONFIG.json'))
 rows=[json.loads(x) for x in open(cfg['candidate_manifest'])];by={r['candidate_id']:r for r in rows};ids=json.load(open(a.ids));runner=old.imp(cfg['frozen_executor'],'smoke_executor_'+str(os.getpid()));ctx=old.Context(cfg);model,tok=runner.load_model(ctx);model.requires_grad_(False);base=runner.base_hash(model)
 ready={'pid':os.getpid(),'label':a.label,'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_memory_allocated_bytes':torch.cuda.memory_allocated(),'gpu':torch.cuda.get_device_name(0)}
 if a.ready:atomic(a.ready,ready)
 if a.start:
  while not Path(a.start).exists():time.sleep(.02)
 started=time.time();out=[]
 for i,cid in enumerate(ids):
  c=by[cid];t=time.time();rec,pred=old.execute(runner,ctx,model,tok,ctx.by_id[c['canonical_sample_id']],c['full_chunk_level_action_sequence'],'worker2_migration_smoke_'+a.label,i+1,len(ids),RUN/'smoke/worker2_parity'/f'{a.label}_progress.json')
  if runner.base_hash(model)!=base:raise RuntimeError('BASE_WEIGHT_MUTATION')
  out.append({'candidate_id':cid,'canonical_sample_id':c['canonical_sample_id'],'score':float(rec['score']),'prediction_sha256':rec['prediction_hash'],'A':c['action_A'],'B':c['action_B'],'tau':c['tau'],'schedule_hash':c['canonical_schedule_hash'],'scorer_sha256':old.SCORER_SHA,'seconds':time.time()-t,'gpu_peak_memory_bytes':rec['gpu_peak_memory_bytes']})
 wall=time.time()-started;atomic(a.output,{'label':a.label,'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'model_idle_memory_bytes':ready['cuda_memory_allocated_bytes'],'execution_wall_seconds':wall,'candidates_per_hour':len(ids)/wall*3600,'results':out})
 del model,tok;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
