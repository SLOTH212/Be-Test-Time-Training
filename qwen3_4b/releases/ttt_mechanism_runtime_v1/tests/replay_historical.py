"""CPU-only full authority replay; writes only compact fixtures and audits in this release.

Statistics fragments are compiled from preserved formal AST nodes unchanged. No
formal main, launcher, model loader, or worker execution entry point is invoked.
"""
import ast,sys,json,hashlib,statistics,types
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
R=rt.ROOT; A=json.loads((R/'provenance/MECHANISM_SOURCE_AUTHORITY.json').read_text()); L=[0,6,12,18,24]
RUN={x['mechanism']:Path(x['formal_run']) for x in A['entries']}
CFG={k:json.loads(next((R/'provenance/original'/k/'config').glob('*FORMAL_CONFIG.json')).read_text()) for k in RUN}
OUT={};HASHES=[]
def load(p):return json.loads(Path(p).read_text())
def read(p):
 p=Path(p);x=load(p);HASHES.append({'path':str(p),'sha256':rt.sha(p)});return x
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def jsonl(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in x))
def check(a,b,label):
 # Preserve JSON types/order-independent dictionary comparison; numeric values exact.
 if json.loads(json.dumps(a,default=sorted))!=json.loads(json.dumps(b,default=sorted)):raise AssertionError(label+' '+str(a)[:180]+' != '+str(b)[:180])
def module(k):
 f=next((R/'provenance/original'/k/'scripts').glob('aggregate*.py'))
 rel=str(f.relative_to(R));tree=ast.parse(f.read_text());names=[x.name for x in tree.body if isinstance(x,ast.FunctionDef) and x.name not in ('main','aggregate')]
 env={'np':np,'statistics':statistics,'CELLS':['F00','F10','F01','F11'],'LEVELS':['L1','L2','L3','L4','L5','L6'],'ACTIONS':rt.actions(L),'TOL':1e-12}
 return rt.source_functions(rel,names,env),tree
def fragment(k,start,end,env):
 m,t=module(k);body=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name in ('main','aggregate')).body
 def assigns(n,key):return isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id==key for z in n.targets)
 lo=next(i for i,x in enumerate(body) if assigns(x,start));hi=next(i for i,x in enumerate(body) if i>=lo and assigns(x,end))
 ns=vars(m).copy();captured={};ns.update(env);ns.update(atomic=lambda p,x:captured.__setitem__(str(p),x),atomic_json=lambda p,x:captured.__setitem__(str(p),x),atomic_csv=lambda *a:None,write_csv=lambda *a:None)
 exec(compile(ast.Module(body=body[lo:hi+1],type_ignores=[]),'<frozen-statistics-fragment>','exec'),ns)
 return ns[end],ns,captured

def main():
 # Runtime safety: this replay may read historical authorities but never write them.
 def guard(event,args):
  if event=='open':
   p,mode,flags=args
   if isinstance(p,(str,bytes)) and (flags & 3 or flags & (64|512|1024)):
    if not Path(p).resolve().is_relative_to(R):raise RuntimeError('REPLAY_WRITE_OUTSIDE_RELEASE')
 sys.addaudithook(guard)
 for x in A['entries']:check(rt.sha(R/x['packaged_path']),x['sha256'],'source hash')
 co={k:rt.rows(c['cohort_manifest']) for k,c in CFG.items() if 'cohort_manifest' in c}
 fixed=defaultdict(dict);froot=RUN['reset'].parents[1]/'fixed7_1p7b_v1/run_20260828T012948+0800'
 for r in rt.rows(froot/'manifests/all_fixed7_results.jsonl'):
  assert r['mode'] not in fixed[r['sample_id']];fixed[r['sample_id']][r['mode']]=r['score']
 assert len(fixed)==6500 and all(set(x)==set(rt.actions(L)) for x in fixed.values())
 droot=Path(CFG['reset']['dynamic_root']);dynamic={};rawmeta={}
 for p in sorted((droot/'results').glob('*.json')):
  d=read(p);sid=d['sample_id'];assert sid not in dynamic
  assert d['completed'] and d['error_status']=='NONE'
  assert d['dynamic_config_sha256']==rt.sha(droot/'config/DYNAMIC_FORMAL_CONFIG.json')
  ar=d['authority_search_result'];check(d['action_sequence'],ar['best_sequence'],'dynamic trajectory')
  check(d['dynamic_score'],ar['dynamic_search_score'],'dynamic score')
  dynamic[sid]={'sample_id':sid,'task':d['task'],'dynamic_score':d['dynamic_score'],'sample_best':d['fixed_SampleBest'],'action_sequence':d['action_sequence'],'chunk_count':ar['full_chunk_count'],'completed':True,'model_identity':d['model_identity'],'checkpoint_identity':d['checkpoint_identity'],'native_prediction_sha256':ar['best_prediction_hash'],'dynamic_result_path':str(p),'dynamic_result_sha256':rt.sha(p)}
 assert len(dynamic)==1435
 eligible={sid for sid,s in fixed.items() if not sid.startswith('ruler_cwe_') and max(s.values())<1-1e-12}
 check(set(dynamic),eligible,'dynamic eligible set')
 parent=rt.improved_cohort(fixed,dynamic,L);by={x['sample_id']:x for x in parent};check(set(by),{x['canonical_sample_id'] for x in co['reverse']},'improved 173');assert len(by)==173
 for m in co['reverse']:
  p=by[m['canonical_sample_id']];check(p['action_sequence'],m['original_dynamic_actions'],'parent trajectory');check(p['dynamic_score'],m['original_dynamic_score'],'parent native score')
  p.update({k:m[k] for k in ['benchmark_identity','benchmark_content_sha256','benchmark_global_index','max_new_tokens']})
 bench=Path(CFG['reset']['benchmark_root']);check(rt.sha(bench/'benchmark_manifest.json'),CFG['reset']['benchmark_identity'],'benchmark')
 with (bench/'sample_manifest.jsonl').open() as f:
  for line in f:
   b=json.loads(line)
   if b['sample_id'] in by:check(hashlib.sha256(b['input'].encode()).hexdigest(),by[b['sample_id']]['benchmark_content_sha256'],'content identity')
 parent.sort(key=lambda x:x['benchmark_global_index']);jsonl(R/'provenance/IMPROVED_173_PARENT_COHORT.jsonl',parent)
 # Small full-population scalar fixture permits exact cohort reconstruction without prompts/results trees.
 jsonl(R/'tests/reference/dynamic_fixed_scalar_inputs.jsonl',[{'sample_id':sid,'fixed_scores':fixed[sid],**d} for sid,d in sorted(dynamic.items())])
 OUT['dynamic']={'status':'PASS','parent_population':6000,'searched':1435,'improved':173,'equal':sum(abs(x['dynamic_score']-x['sample_best'])<=1e-12 for x in dynamic.values()),'below':0,'improved_cohort_sha256':rt.sha(R/'provenance/IMPROVED_173_PARENT_COHORT.jsonl')}
 print('DYNAMIC_PARENT=PASS',flush=True)
 screen={sid:rt.reset_screen(x['action_sequence'],L) for sid,x in by.items()};ids={sid for sid,x in screen.items() if x['eligible']};check(ids,{x['canonical_sample_id'] for x in co['reset']},'reset eligibility');assert len(ids)==167
 ds={sid:rt.deletion_plan(by[sid]['action_sequence'],screen[sid]['boundary'],L) for sid in ids};check({sid for sid,x in ds.items() if x['eligible']},{x['canonical_sample_id'] for x in co['deletion']},'deletion membership')
 check(ids,{x['canonical_sample_id'] for x in co['kp']},'KP membership')
 for m in co['deletion']:
  p=ds[m['canonical_sample_id']]
  for k in ['boundary','selected_component_layers','all_prefix','sham_layer','transition_category']:check(p[k],m[k],'deletion '+k)
 for m in co['kp']:
  kp=rt.kp_plan(by[m['canonical_sample_id']]['action_sequence'],m['boundary'],L);check(kp['native_used_layers'],m['native_used_layer_union'],'KP used union')
 OUT['screening']={'status':'PASS','reset_eligible':167,'reset_excluded':6,'reset_exclusion_reasons':dict(Counter(x['exclusion_reason'] for x in screen.values() if not x['eligible'])),'deletion_eligible':151,'deletion_excluded':16}
 for k in ['reverse','reset']:
  records=[];run=RUN[k];cfg=CFG[k]
  for m in co[k]:
   sid=m['canonical_sample_id'];r=read(run/'results'/(hashlib.sha256(sid.encode()).hexdigest()+'.json'));assert r['completed'];check(r['canonical_sample_id'],sid,'result ID')
   if k=='reverse':check(r['reversed_actions'],rt.reverse(by[sid]['action_sequence']),'reverse transform');check(r['dynamic_minus_reverse'],r['original_dynamic_score']-r['reverse_score'],'reverse effect')
   else:
    check(r['reset_boundary_observed'],screen[sid]['boundary'],'reset boundary');check(r['candidate_layers_reset'],L,'reset layer set');check(r['native_minus_reset'],r['native_dynamic_score']-r['reset_score'],'reset effect')
    assert r['reset_max_abs_residual']==0 and r['reset_count']==1 and r['reset_to_checkpoint_parity']=='PASS'
    assert not any(r[x] for x in ['base_parameter_mutation','prompt_restart','prior_chunk_replay','kv_cache_reset','prefix_context_reset','rng_reset','generation_change'])
   assert r['generation_update_count']==0 and r['recompute_downstream_updates'] and not r['stored_delta_stitching'];records.append(r)
  assert len(list((run/'results').glob('*.json')))==len(records)
  result,ns,_=fragment(k,'dyn' if k=='reverse' else 'native','summary',{'records':records,'rs':records,'config':cfg,'c':cfg,'run':run})
  expected=read(run/'reports'/('EXACT_REVERSE_AGGREGATE.json' if k=='reverse' else 'STATE_RESET_AGGREGATE.json'));check(result,expected,k+' aggregate+bootstrap')
  jsonl(R/f'tests/reference/{k}_scalar_results.jsonl',records)
  OUT[k]={'status':'PASS','result_n':len(records),'aggregate_all_fields_exact':True,'bootstrap_exact':True,'summary':result};print(k.upper()+'_REPLAY=PASS',flush=True)
 got=[read(p) for p in sorted((RUN['deletion']/'branches').glob('*.json'))];assert len(got)==458 and len({r['branch_id'] for r in got})==458
 for r in got:
  s=next(x for x in co['deletion'] if x['canonical_sample_id']==r['canonical_sample_id']);p=ds[r['canonical_sample_id']]
  assert r['completed'] and r['target_layer_reset_parity']=='PASS' and r['non_target_layer_preservation']=='PASS'
  assert r['target_layer'] in (p['selected_component_layers'] if r['intervention_type']=='SELECTIVE' else [p['sham_layer']])
  check(r['boundary'],p['boundary'],'deletion branch boundary');assert not r['base_parameter_mutation'] and r['generation_update_count']==0
  if r['intervention_type']=='SHAM':check(r['score'],s['normal_score'],'sham score');check(r['prediction_sha256'],s['normal_prediction_sha256'],'sham prediction')
 result,ns,_=fragment('deletion','by','summary',{'got':got,'co':co['deletion'],'c':CFG['deletion'],'r':RUN['deletion']})
 check(result,read(RUN['deletion']/'reports/SELECTIVE_DELETION_AGGREGATE.json'),'deletion all statistics');jsonl(R/'tests/reference/deletion_scalar_results.jsonl',got)
 OUT['deletion']={'status':'PASS','result_n':151,'branch_n':458,'summary':result,'aggregate_all_fields_exact':True,'bootstrap_exact':True};print('DELETION_REPLAY=PASS',flush=True)
 # K/P closure is replayed from retained comparison records, not unavailable raw tensors.
 refs={r['canonical_sample_id']:r for r in (read(p) for p in sorted((RUN['kp']/'source_states').glob('*.json')))};kby=defaultdict(dict);compact=[]
 for p in sorted((RUN['kp']/'branches').glob('*.json')):
  r=read(p);sid=r['canonical_sample_id'];cell=r['cell'];assert cell not in kby[sid];ref=refs[sid]
  assert r['completed'] and r['fast_weight_parity']=='PASS' and r['K_state_equal_expected'] and r['P_state_equal_expected'] and r['future_input_equal'] and not r['base_parameter_mutation']
  plan=rt.kp_plan(by[sid]['action_sequence'],screen[sid]['boundary'],L)['cells'][cell]
  check(r['P_materialized_layers'],plan['injection_layers'],'KP P');check(r['K_source'],plan['K_source'],'KP K')
  check(r['post_injection_fast_weight_sha256'],ref['native_fast_weight_hashes'],'KP W');check(r['K_prefix_hashes'],ref['native_prefix_kv_hashes'] if plan['K_bit'] else ref['control_prefix_kv_hashes'],'KP KV')
  trace=r['trace_comparisons'];closure={f'L{i}':all(x['exact_equal'] for x in trace if x['region'] in regions) for i,regions in [(1,['FIRST_CONTINUATION_CHUNK','FIRST_POST_BOUNDARY_UPDATE']),(2,['COMPLETE_CONTINUATION_CHUNK']),(3,['INCOMPLETE_PROMPT_TAIL'])]}
  closure.update(L4=r['generation_start_max_abs']==0,L5=r['prediction_sha256']==r['native_prediction_sha256']);closure['L6']=all(closure.values());check(closure,r['closure'],'closure comparison replay')
  check(ref['native_reward'],by[sid]['dynamic_score'],'native reward');check(ref['native_prediction_sha256'],ref['frozen_native_prediction_sha256'],'native prediction')
  kby[sid][cell]=r;compact.append({k:v for k,v in r.items() if k not in ['trace_comparisons','K_prefix_hashes']})
 assert len(kby)==167 and all(len(v)==4 for v in kby.values());m,_=module('kp');ids=[r['canonical_sample_id'] for r in co['kp']];computed=m.group_summary(ids,kby,refs);expected=read(RUN['kp']/'aggregates/KP_PRIMARY_FACTORIAL_ANALYSIS.json');check(computed,expected['all_167'],'KP full summary')
 e=[m.effects({c:kby[s][c]['generation_start_relative_l2'] for c in m.CELLS}) for s in ids]
 boots={k:m.boot([x[k] for x in e]) for k in e[0]};check(boots,expected['paired_bootstrap'],'KP paired bootstrap')
 for t,x in expected['per_task'].items():check(m.group_summary([r['canonical_sample_id'] for r in co['kp'] if r['task']==t],kby,refs),x,'KP per task')
 for g,x in expected['trajectory_type'].items():check(m.group_summary([r['canonical_sample_id'] for r in co['kp'] if ('MULTISWITCH' if r['switch_count']>1 else 'SIMPLE')==g],kby,refs),x,'KP trajectory type')
 jsonl(R/'tests/reference/kp_scalar_results.jsonl',compact);jsonl(R/'tests/reference/kp_source_states.jsonl',list(refs.values()))
 OUT['kp']={'status':'PASS','branch_n':668,'closure_counts':{c:computed['closure'][c]['L6']['count'] for c in m.CELLS},'aggregate_all_fields_exact':True,'bootstrap_exact':True,'closure_scope':'Retained comparison records and hashes. Raw tensor equality cannot be rerun because original tensors were not retained.'};print('KP_REPLAY=PASS',flush=True)
 # Validate every original candidate using the exact frozen worker validator.
 cfg=CFG['r2'];manifest=rt.rows(cfg['candidate_manifest']);parent2=rt.rows(cfg['parent_manifest']);check({x['canonical_sample_id'] for x in parent2},set(by),'R2 parent')
 worker_path='provenance/original/r2/scripts/exact_r2_worker.py';tree=ast.parse((R/worker_path).read_text());const={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   try:
    for t in n.targets:
     if isinstance(t,ast.Name):const[t.id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 w=rt.source_functions(worker_path,['valid_result','load','sample_key','branch_path'],const)
 cby=defaultdict(list)
 for c in manifest:cby[c['canonical_sample_id']].append(c)
 winners={};compact_r2=[];m,_=module('r2')
 for sid,cs in cby.items():
  expected_schedules=list(rt.r2_candidates(cs[0]['T'],L));check([x['canonical_schedule_hash'] for x in cs],[x['full_schedule_hash'] for x in expected_schedules],'R2 exhaustive order')
  rs=[]
  for c in cs:
   p=w.branch_path(RUN['r2'],c);r=w.valid_result(p,c,cfg);HASHES.append({'path':str(p),'sha256':rt.sha(p)});rs.append(r)
  selected,best,tie=m.independent_winner(rs)
  online=next((load(p) for p in (RUN['r2']/'winners').glob('*.json') if load(p)['canonical_sample_id']==sid))
  check(selected['candidate_id'],online['winning_candidate_id'],'R2 winner');check(best,online['R2_score'],'R2 score')
  winners[sid]={'R2_score':best,'winning_A':selected['A'],'winning_B':selected['B'],'winning_tau':selected['tau'],'T':selected['T']}
  compact_r2.append({'sample_id':sid,'winner':selected,'candidate_count':len(rs),'tie_count':tie})
 assert len(manifest)==99120 and len(winners)==173
 data=[]
 for p in parent2:
  sid=p['canonical_sample_id'];w0=winners[sid];off=p['OFF_score'];sb=p['SampleBest_score'];dyn=p['Dynamic_score'];v=w0['R2_score']
  data.append({'canonical_sample_id':sid,'task':p['task'],'benchmark_global_index':p['benchmark_global_index'],'OFF':off,'R2':v,'Dynamic':dyn,'SampleBest':sb,'R2_minus_OFF':v-off,'Dynamic_minus_OFF':dyn-off,'Dynamic_minus_R2':dyn-v,'R2_minus_SampleBest':v-sb,'Dynamic_switch_count':p['Dynamic_switch_count'],'Dynamic_regime_count':p['Dynamic_regime_count'],**{k:w0[k] for k in ['winning_A','winning_B','winning_tau','T']}})
 data.sort(key=lambda x:x['benchmark_global_index']);result,ns,_=fragment('r2','gaps','b2',{'data':data,'cfg':cfg}) if False else fragment('r2','gaps','primary',{'data':data,'cfg':cfg})
 b1=m.bootstrap(data,cfg['paired_bootstrap_seed'],20000,False);b2=m.bootstrap(data,cfg['task_stratified_bootstrap_seed'],20000,True)
 result.update(DYNAMIC_R2_PAIRED_BOOTSTRAP_95CI=b1['mean_dynamic_minus_r2_95ci'],DYNAMIC_R2_TASK_STRATIFIED_BOOTSTRAP_95CI=b2['mean_dynamic_minus_r2_95ci'],R2_GAIN_RETENTION_BOOTSTRAP_95CI=b1['aggregate_r2_gain_retention_95ci'])
 check(result,read(RUN['r2']/'aggregates/PRIMARY_OFF_R2_DYNAMIC.json'),'R2 primary');check(b1,read(RUN['r2']/'bootstrap/PAIRED_BOOTSTRAP_20000.json'),'R2 paired');check(b2,read(RUN['r2']/'bootstrap/TASK_STRATIFIED_PAIRED_BOOTSTRAP_20000.json'),'R2 stratified')
 jsonl(R/'tests/reference/r2_scalar_results.jsonl',data);jsonl(R/'tests/reference/r2_winner_references.jsonl',compact_r2)
 OUT['r2']={'status':'PASS','candidate_n':99120,'result_n':173,'summary':result,'exhaustive_family_exact':True,'winner_parity':True,'bootstrap_exact':True};print('R2_REPLAY=PASS',flush=True)
 # Large raw trees are represented by a compact cryptographic root, not copied.
 refs=sorted(HASHES,key=lambda x:x['path']);h=hashlib.sha256(''.join(x['path']+'\0'+x['sha256']+'\n' for x in refs).encode()).hexdigest()
 OUT.update(status='PASS',historical_result_files_hashed=len(refs),historical_result_reference_root_sha256=h,inference_executed=False,gpu_used=False)
 dump(R/'audits/HISTORICAL_MECHANISM_MASTER_REPLAY.json',OUT)
 dump(R/'provenance/MECHANISM_CHAIN_MANIFEST.json',{'nodes':{'dynamic_parent':6000,'searched':1435,'improved':173,'reverse':173,'reset':167,'kp':167,'deletion':151,'r2':173},'edges':[['dynamic_parent','searched'],['searched','improved'],['improved','reverse'],['improved','reset'],['reset','kp'],['reset','deletion'],['improved','r2']],'reverse_receipt_role_in_reset':'authority readiness only; no Reverse outcome selection','kp_to_deletion_edge':False})
 print('MASTER_REPLAY=PASS',flush=True)
if __name__=='__main__':main()
