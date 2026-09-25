"""Independently verify all aggregate reporting and actual observed scalar values."""
import json, math, hashlib, time
from pathlib import Path
import numpy as np
R=Path('/path/to/ttt');A=R/'audits/v3';D=A/'analysis'
read=lambda p:json.loads(Path(p).read_text())
rows=[json.loads(x) for x in (D/'ALL_TENSOR_PAIR_METRICS.jsonl').read_text().splitlines()];inp=read(A/'REPLICATE_INPUT_PARITY.json');inf=read(D/'ALL_TENSOR_INFERENCE.json');stat=read(A/'STATISTICAL_RESULT_V3.json');base=read(A/'UNINTERRUPTED_NONDETERMINISM_BASELINE.json');res=read(A/'RESUMED_NUMERICAL_VARIABILITY.json');pairs=read(D/'PAIR_METRIC_MANIFEST.json')['pairs'];checks=[]
def ck(name,v):
 checks.append({'check':name,'pass':bool(v)})
 if not v:raise AssertionError(name)
def near(a,b):return np.allclose(a,b,rtol=2e-12,atol=1e-18)
inventory={x['group']+x['canonical_state_name']:x for x in read(A/'V2_FAILED_76_TENSOR_INVENTORY.json')['rows']}
ck('76 original shapes and dtypes',all(x['shape']==inventory[x['key']]['shape'] and x['dtype']==inventory[x['key']]['dtype'] for x in rows if x['primary76']))
ck('complete serialized model and optimizer element counts',sum(x['numel'] for x in rows if x['group']=='model')==4450838016 and sum(x['numel'] for x in rows if x['group']=='optimizer')==8123610522)
for cls,source in [('UU',base['group_aggregates']),('RU',res['RU']),('RR',res['RR'])]:
 indices=[i for i,(l,r) in enumerate(pairs) if ('RU' if l[0]!=r[0] else l[0]+r[0])==cls]
 for group in ['model','optimizer','optimizer_moments','optimizer_step']:
  selected=[row for row in rows if row['group']==('model' if group=='model' else 'optimizer') and (not row['key'].endswith('.step') if group=='optimizer_moments' else row['key'].endswith('.step') if group=='optimizer_step' else True)];values={k:[] for k in ['max_abs','mean_abs','L1','L2','left_reference_relative_L2']}
  for i in indices:
   left,right=pairs[i];n=sum(x['numel'] for x in selected);l1=math.fsum(x['pairs'][i]['L1'] for x in selected);l2=math.sqrt(math.fsum(x['pairs'][i]['L2']**2 for x in selected));norm=math.sqrt(math.fsum(x['run_stats'][left]['norm']**2 for x in selected));v=[max(x['pairs'][i]['max_abs'] for x in selected),l1/n,l1,l2,l2/max(norm,np.finfo(float).tiny)]
   for key,val in zip(values,v):values[key].append(val)
  for key,v in values.items():ck(cls+group+key+' aggregate quantiles',near(np.quantile(v,[.5,.95,1]),[source[group]['distribution'][key][q] for q in ['median','p95','max']]))
for name in inp['names']:
 paths=[R/'runs/qwen3_4b_stage1_resume_v3/attempt2'/name] if name[0]=='U' else [R/'runs/qwen3_4b_stage1_resume_v3/attempt2'/(name+'_pre'),R/'runs/qwen3_4b_stage1_resume_v3/attempt2'/name]
 for p in paths:
  for rank in [0,1]:
   for file in [p/f'load_completeness_rank{rank}.json',p/f'final_rank{rank}.json',p/f'boundary_save_rank{rank}.json']:
    if file.exists():
     lifecycle=read(file)['FSDP'];ck(str(file)+' FSDP idle boundary',len(lifecycle)==37 and all(x['state']=='TrainingState.IDLE' and x['param_group_state']=='TrainingState.IDLE' and x['post_forward_order_n'] in [None,0] and x['reduce_scatter_pending'] in [None,False] for x in lifecycle))
  norms=[{x['update_step']:x['local_unique_parameter_squared_norm'] for x in map(json.loads,(p/f'parameter_norm_rank{rank}.jsonl').read_text().splitlines())} for rank in [0,1]]
  for line in map(json.loads,(p/'training.jsonl').read_text().splitlines()):
   step=line['update_step'];obs=next(x for x in inp['trajectories'][name] if x['step']==step)
   ck(name+str(step)+' actual loss and norms',obs['loss']==line['global_mean_loss'] and obs['gradient_norm_before_clip']==line['grad_norm_before_clip'] and obs['gradient_norm_after_clip']==line['grad_norm_after_clip'] and obs['parameter_norm']==math.sqrt(norms[0][step]+norms[1][step]))
abnormal=any(x['confirmed_abnormal'] for x in inf);passing=all(x['empirical_equivalence_pass'] for x in inf);case='CASE_C' if abnormal else 'CASE_B_CANDIDATE' if passing else 'CASE_D'
ck('overall classification follows recomputed tensor gates',stat['statistical_case']==case)
ck('all conditional localization triggers retained',stat['localization_required']==[x['key'] for x in inf if x['localization_trigger']])
ck('counts recomputed',all(stat['counts'][g][k]==sum(x['group']==g and (True if k=='total' else x['exact_all_pairs'] if k=='exact' else x['legacy_allclose_all_pairs'] if k=='legacy_allclose' else x['empirical_equivalence_pass']) for x in inf) for g in ['model','optimizer','loss'] for k in ['total','exact','legacy_allclose','empirical']))
receipts=read(A/'REPLICATE_CONTROLLER_RECEIPTS.json')+read(A/'execution2/REPLICATE_CONTROLLER_RECEIPTS.json');cost=read(A/'DIAGNOSTIC_COST_BREAKDOWN.json')
ck('all-attempt GPU cost accounts for invalid observer attempt',sum(x['optimizer_updates_observed'] for x in receipts)==inp['all_attempt_optimizer_updates']==cost['all_attempt_optimizer_updates'] and sum(x['wall_seconds'] for x in receipts)==inp['all_attempt_torchrun_wall_seconds']==cost['all_attempt_torchrun_wall_seconds'])
iteration=0.0
for x in receipts:
 p=Path(x['output'])/'training.jsonl'
 if p.exists():iteration+=sum(json.loads(line)['iteration_seconds_excluding_checkpoint'] for line in p.read_text().splitlines())
ck('iteration time breakdown recomputes',iteration==cost['logged_iteration_seconds_excluding_checkpoint'])
optimization=A/'METRICS_CPU_EQUIVALENT_OPTIMIZATION.json'
if optimization.exists():
 op=read(optimization);archived=Path(op['archived_original_output']).read_bytes();ck('all original measured rows preserved unchanged',(D/'ALL_TENSOR_PAIR_METRICS.jsonl').read_bytes().startswith(archived))
 proof=read(A/'CACHED_TOLERANCE_REAL_TENSOR_EQUIVALENCE.json');ck('cached tolerance actual full-tensor equivalence proof',proof['status']=='PASS' and proof['pair_count']==45 and proof['optimized_source_sha256']==op['optimized_source_sha256'])
payload={'status':'PASS','scope':'All60 group aggregate quantile distributions, actual loss/gradient/parameter norm observations, whole-state counts, overall statistical classification and conditional-localization list independently recomputed.','checks':checks,'check_count':len(checks),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed_at':time.time()}
(A/'REPORTING_INDEPENDENT_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print('REPORTING_INDEPENDENT_AUDIT_PASS',len(checks))
