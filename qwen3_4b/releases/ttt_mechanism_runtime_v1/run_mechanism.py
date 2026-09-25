#!/usr/bin/env python3
"""Build cohorts, plan, or explicitly execute one mechanism after Fixed + Dynamic."""
import argparse,json,sys
from pathlib import Path
import runtime as rt
from worker import Store,atomic

def inputs(cfg):
 fixed={}
 for r in rt.rows(cfg['fixed_results']):
  sid=r['sample_id'];a=r.get('mode',r.get('action'))
  if a in fixed.setdefault(sid,{}):raise ValueError('DUPLICATE_FIXED_RESULT')
  fixed[sid][a]=r['score']
 p=Path(cfg['dynamic_results']);records=(json.loads(f.read_text()) for f in sorted(p.glob('*.json'))) if p.is_dir() else iter(rt.rows(p));dynamic={}
 for r in records:
  sid=r['sample_id']
  if sid in dynamic:raise ValueError('DUPLICATE_DYNAMIC_RESULT')
  ar=r.get('authority_search_result',{})
  dynamic[sid]={'sample_id':sid,'task':r['task'],'dynamic_score':r['dynamic_score'],'sample_best':r.get('fixed_SampleBest',r.get('sample_best')),'action_sequence':r['action_sequence'],'chunk_count':r.get('trajectory_length',r.get('chunk_count',ar.get('full_chunk_count'))),'completed':r.get('completed',r.get('search_status')=='SUCCESS'),'native_prediction_sha256':ar.get('best_prediction_hash',r.get('prediction_hash'))}
 parents=rt.improved_cohort(fixed,dynamic,cfg['configured_ttt_layers']);selected=[]
 index={}
 with (Path(cfg['benchmark_root'])/'sample_manifest.jsonl').open() as f:
  for i,line in enumerate(f):index[json.loads(line)['sample_id']]=i
 for p in parents:p['benchmark_global_index']=index[p['sample_id']]
 parents.sort(key=lambda p:p['benchmark_global_index'])
 for p in parents:
  mech=cfg['mechanism'];scr=rt.reset_screen(p['action_sequence'],cfg['configured_ttt_layers'])
  if mech in ['reset','deletion','kp'] and not scr['eligible']:continue
  if mech=='deletion' and not rt.deletion_plan(p['action_sequence'],scr['boundary'],cfg['configured_ttt_layers'])['eligible']:continue
  selected.append(p)
 if cfg['mechanism']=='deletion':
  if 'reset_results' not in cfg:raise ValueError('DELETION_REQUIRES_MATCHED_RESET_RESULTS')
  reset={}
  for f in sorted((Path(cfg['reset_results'])/'samples').glob('*.json')):
   r=json.loads(f.read_text());sid=r['sample_id'];body={k:v for k,v in r.items() if k!='payload_sha256'}
   if rt.canonical(body)!=r.get('payload_sha256') or r['status']!='COMMITTED' or r['intervention']!='reset':raise ValueError('INVALID_RESET_RESULT')
   if r['model_identity']!=cfg['model_identity'] or r['benchmark_identity']!=cfg['benchmark_identity']:raise ValueError('RESET_IDENTITY')
   reset[sid]=r
  for p in selected:
   r=reset[p['sample_id']]
   if r['parent_dynamic_trajectory']!=p['action_sequence'] or r['parent_dynamic_score']!=p['dynamic_score']:raise ValueError('RESET_PARENT_PARITY')
   p['full_reset_score']=r['intervention_score']
 return fixed,parents,selected

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--execute',action='store_true');a.add_argument('--workers',type=int,default=1);a.add_argument('--worker-id',type=int,default=0);args=a.parse_args();cfg=rt.load_config(args.config)
 if args.workers<1 or not 0<=args.worker_id<args.workers:raise ValueError('WORKER_SHARD')
 dep=rt.dependency()
 if not args.execute:
  result={'status':'STATIC_PLAN','mechanism':cfg['mechanism'],'actions':cfg['actions'],'chunk_size':cfg['chunk_size'],'complete_chunk_capacity':cfg['context_length']//cfg['chunk_size'],'future_cohort_frozen':False,'inference_executed':False}
  if cfg['fixed_results']!='FIXED_RESULTS_JSONL' and cfg['dynamic_results']!='DYNAMIC_RESULTS_JSONL':
   _,p,s=inputs(cfg);result.update(improved_n=len(p),mechanism_n=len(s),cohort_sha256=rt.canonical(s))
  print(json.dumps(result,sort_keys=True));return
 if cfg['mechanism']=='template':raise ValueError('SELECT_ONE_MECHANISM')
 from execution import Executor
 fixed,parents,selected=inputs(cfg);code_hashes={p.name:rt.sha(p) for p in rt.ROOT.glob('*.py')}
 identity={'config':cfg,'cohort_sha256':rt.canonical(selected),'cohort_n':len(selected),'runtime_sources':code_hashes,'source_authority_sha256':rt.sha(rt.ROOT/'provenance/MECHANISM_SOURCE_AUTHORITY.json'),'inference_dependency':json.loads((rt.ROOT/'provenance/INFERENCE_RUNTIME_DEPENDENCY.json').read_text())}
 store=Store(cfg['run_root'],identity);owned=[x for i,x in enumerate(selected) if i%args.workers==args.worker_id]
 # Each worker has its own immutable evaluator workspace; only final samples are shared.
 workspace=Path(cfg['run_root'])/'workers'/str(args.worker_id);workspace.mkdir(parents=True,exist_ok=True)
 ex=None
 for p in owned:
  sid=p['sample_id']
  with store.own(sid):
   if store.committed(sid) is not None:continue
   if ex is None:
    bench=rt.rows(Path(cfg['benchmark_root'])/'sample_manifest.jsonl')
    for i,r in enumerate(bench):r['_sample_index']=i
    ex=Executor(cfg,bench,fixed,workspace)
   result=rt.validate_result(ex.execute(p))
   if cfg['mechanism']=='deletion':result['details']['deletion']['full_reset_score']=p['full_reset_score']
   store.commit(sid,result)
 print(json.dumps({'worker_id':args.worker_id,'owned_samples':len(owned),'status':'COMPLETE'}))
if __name__=='__main__':main()
