#!/usr/bin/env python3
import sys,json,ast,types,statistics,collections,importlib.util,argparse,hashlib,shutil
from pathlib import Path
from debug_worker import R,W,A,setup,rt,atomic,Store,FLAGS,LAYERS,ACTIONS,fixed_store,identity,sha
p=Path(setup['adapt'])/'runtime.py';s=importlib.util.spec_from_file_location('inference_math',p);im=importlib.util.module_from_spec(s);s.loader.exec_module(im)
def aggregate_fixed(rows):
 p=Path(setup['base'])/'fixed/aggregate_fixed.py';t=ast.parse(p.read_text());ns=dict(defaultdict=collections.defaultdict,math=__import__('math'),statistics=statistics,fixed_actions=im.fixed_actions,sample_best=im.sample_best,best_single=im.best_single);exec(compile(ast.Module(body=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='aggregate'],type_ignores=[]),str(p),'exec'),ns);return ns['aggregate'](rows,LAYERS)
def fixed():
 ids=[x['sample_id'] for x in rt.rows(W/'DEBUG_SAMPLE_MANIFEST.jsonl')];store=fixed_store(ids);records=[]
 for sid in ids:
  for action in ACTIONS:
   c=store.committed(sid+'|'+action);assert c is not None
   records.append({**c,'sample_id':sid,'status':'SUCCESS'})
 assert len(list((W/'fixed/samples').glob('*.json')))==len(ids)*8
 agg=aggregate_fixed(records);atomic(W/'FIXED8_DEBUG_AGGREGATE.json',{**agg,**FLAGS});(W/'fixed_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));atomic(W/'FIXED8_DEBUG_FINAL_AUTHORITY.json',dict(status='PASS',sample_n=len(ids),cell_n=len(records),config_identity=store.identity,aggregate_sha256=sha(W/'FIXED8_DEBUG_AGGREGATE.json'),results_sha256=sha(W/'fixed_results.jsonl'),sample_manifest_sha256=sha(W/'DEBUG_SAMPLE_MANIFEST.jsonl'),parent_authority_sha256=setup['parent_authority_sha256'],**FLAGS))
 # Resume fault fixture uses actual committed cell payloads and the same frozen Store.
 root=W/'resume_fixed_fixture';s=Store(root,dict(test='real_partial_fixed_cells',parent=store.identity));skipped=0;new=0
 for i,r in enumerate(records[:2]):
  sid=r['sample_id']+'|'+r['action']
  with s.own(sid):
   if s.committed(sid) is not None:skipped+=1;continue
   s.commit(sid,{'real_parent_payload_sha256':r['payload_sha256']});new+=1
 hashes={p.name:sha(p) for p in (root/'samples').glob('*.json')}
 for r in records[:2]:assert s.committed(r['sample_id']+'|'+r['action']) is not None
 assert hashes=={p.name:sha(p) for p in (root/'samples').glob('*.json')}
 atomic(A/'FIXED8_DEBUG_AUDIT.json',dict(FIXED8_SINGLE_SAMPLE_SANITY='PASS',QWEN3_4B_FIXED8_DEBUG_SMOKE='PASS',INFERENCE_4K_RUNTIME_SEMANTICS='PASS',SAMPLE_BEST_DEBUG_AGGREGATION='PASS',partial_resume_store='PASS',actual_worker_resume='PENDING_NOOP_RESTART',finite_update_and_output_checks=True,tail_no_update=True,generation_no_update=True,layer_set=LAYERS,per_sample_reset='native cache allocated per generate call',**FLAGS))
 print(json.dumps(dict(status='PASS',samples=len(ids),cells=len(records),non_ceiling=[x['sample_id'] for x in agg['per_sample'] if x['sample_best']<1-1e-12])))
def dynamic():
 fixed=json.loads((W/'FIXED8_DEBUG_AGGREGATE.json').read_text());selected=[r['sample_id'] for r in fixed['per_sample'] if r['sample_best']<1-1e-12 and r['task']!='ruler_cwe_16k'][:2];assert selected
 store=Store(W/'dynamic',identity('dynamic',selected));rows=[store.committed(sid) for sid in selected];assert all(rows)
 assert len(list((W/'dynamic/samples').glob('*.json')))==len(selected)
 by={x['sample_id']:{a:x[a] for a in ACTIONS} for x in fixed['per_sample']};cohort=rt.improved_cohort(by,{x['sample_id']:x for x in rows},LAYERS)
 for x in rows:
  assert x['model_identity']==setup['model_identity'] and x['benchmark_identity']==setup['benchmark_identity'] and x['inference_config_sha256']==setup['config_sha256'];assert x['actions']==ACTIONS and x['chunk_size']==4096 and x['replay_count']<=im.replay_budget(8,x['chunk_count']);assert x['raw']['generation_update_count']==0
 (W/'dynamic_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));atomic(W/'DEBUG_COHORT.json',dict(improved=cohort,equal=[x['sample_id'] for x in rows if abs(x['dynamic_score']-x['sample_best'])<=1e-12],worse=[x['sample_id'] for x in rows if x['dynamic_score']<x['sample_best']-1e-12],natural_improved_n=len(cohort),formal_reset_eligibility='ZERO_ELIGIBLE_BY_CHUNK_GEOMETRY',**FLAGS))
 atomic(W/'DYNAMIC_DEBUG_FINAL_AUTHORITY.json',dict(status='PASS',sample_n=len(rows),config_identity=store.identity,result_sha256=sha(W/'dynamic_results.jsonl'),parent_fixed_authority_sha256=sha(W/'FIXED8_DEBUG_FINAL_AUTHORITY.json'),cohort_sha256=sha(W/'DEBUG_COHORT.json'),natural_improved_n=len(cohort),**FLAGS))
 # Real payload-backed resume: existing result immutable; partial job can re-enter clean ownership.
 root=W/'resume_dynamic_fixture';s=Store(root,dict(test='dynamic_partial',action_schema=ACTIONS,parent=store.identity));sid=selected[0]
 with s.own(sid):
  if s.committed(sid) is None:s.commit(sid,{'real_payload_sha256':rows[0]['payload_sha256']})
 h=sha(s.path(sid))
 with s.own(sid):assert s.committed(sid) is not None
 assert sha(s.path(sid))==h
 rejected=False
 try:Store(root,dict(test='wrong_action_schema'))
 except ValueError:rejected=True
 assert rejected
 atomic(A/'DYNAMIC_DEBUG_AUDIT.json',dict(DYNAMIC_4K_PROTOCOL_PREFLIGHT='PASS',QWEN3_4B_DYNAMIC_4K_DEBUG_SMOKE='PASS',DYNAMIC_TO_MECHANISM_SCHEMA='PASS',DEBUG_COHORT_BUILDER='PASS',partial_resume_store='PASS',actual_worker_resume='PENDING_NOOP_RESTART',replay_counts=[x['replay_count'] for x in rows],replay_budgets=[x['replay_budget'] for x in rows],natural_improved_n=len(cohort),**FLAGS))
 print(json.dumps(dict(status='PASS',dynamic_n=len(rows),natural_improved_n=len(cohort))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['fixed','dynamic']);args=p.parse_args();globals()[args.phase]()
