import json,hashlib,math,time,collections
from pathlib import Path
import torch
R=Path('/path/to/ttt');L=R/'audits/v3/localization_final';RUN=R/'runs/qwen3_4b_stage1_resume_v3/localization_final';N=R/'work/qwen3_4b_stage1_resume_v3/localization_final'
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
def ck(name,test):
 checks.append({'check':name,'pass':bool(test)})
 if not test:raise AssertionError(name)
def near(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-20)
plan=read(L/'CONDITIONAL_LOCALIZATION_PLAN.json');source_cpu=read(R/'audits/qwen3_4b_stage1_resume_equivalence_v3/CPU_REVIEW_PENDING_LOCALIZATION_AUDIT.json');metrics=read(L/'LOCALIZATION_METRICS.json');summary=read(L/'LOCALIZATION_SUMMARY.json');targets=read(L/'PER_TARGET_LOCALIZATION.json');receipts=read(L/'LOCALIZATION_RUN_RECEIPTS.json');obs=read(L/'OBSERVER_READ_ONLY_AUDIT.json')
for p,h in read(L/'PRIOR_EVIDENCE_IMMUTABILITY.json')['files'].items():ck('Prior file unchanged '+p,sha(Path(p))==h)
ck('exact48 original targets',[x['target'] for x in targets]==source_cpu['statistical_result']['localization_required'] and len(targets)==48)
ck('observer source and CPU semantic audit bound',sha(N/'entry.py')==plan['observer_sha256']==obs['observer_sha256'] and obs['status']=='PASS' and all(obs['checks'].values()))
ck('three valid phases and six updates',len(receipts)==3 and all(x['status']=='VALID_COMPLETED_PHASE' for x in receipts) and sum(x['optimizer_updates_observed'] for x in receipts)==6)
for item in receipts:
 ck('same frozen config and GPU allowlist '+item['role'],item['environment']['CUDA_VISIBLE_DEVICES']=='2,3' and item['environment']['WORLD_SIZE']=='2' and item['command'][item['command'].index('--config')+1]==plan['config'] and item['observer_sha256']==plan['observer_sha256'])
ck('config unchanged',sha(Path(plan['config']))==plan['config_sha256'])
metas={name:[read(RUN/name/f'final_rank{r}.json') for r in (0,1)] for name in ['U_LOC','R_LOC']};raw={name:[torch.load(RUN/name/f'final_rank{r}.pt',mmap=True,weights_only=True,map_location='cpu') for r in (0,1)] for name in metas}
for rank in (0,1):
 saved=read(RUN/'R_LOC_pre/checkpoints'/f'presave_step2_rank{rank}.json');loaded=read(RUN/'R_LOC'/f'boundary_contract_rank{rank}.json');uc=read(RUN/'U_LOC'/f'boundary_contract_rank{rank}.json')
 ck('exact saved vs fresh-loaded preforward rank'+str(rank),saved==loaded)
 ck('U R input RNG scheduler groups parity rank'+str(rank),all(uc[k]==saved[k] for k in ['rank_state','param_groups','buffers','runtime']))
 for name in metas:
  m=metas[name][rank];ck(name+' runtime original rank'+str(rank),m['runtime']['CUDA_VISIBLE_DEVICES']=='2,3' and m['runtime']['WORLD_SIZE']==2 and m['runtime']['deterministic'] is False and m['observer_read_checks']>0)
  ck(name+' final step3 rank'+str(rank),m['progress']['update_step']==3 and m['progress']['record_cursor']==6)
# Independently recompute every full-vector metric using torch reductions.
for index,row in enumerate(metrics['rows']):
 key=row['key']
 if not row['available']:
  ck(key+' absence accurately reported',all(row['presence'][name][rank]==(key in raw[name][rank]) for name in raw for rank in (0,1)));continue
 ranks=(0,) if metas['U_LOC'][0]['tensors'][key]['replicated'] and not key.startswith('pre_reduce.') else (0,1)
 mx=total=sq=unorm=rnorm=0.;n=nz=0
 for rank in ranks:
  a=raw['U_LOC'][rank][key].reshape(-1);b=raw['R_LOC'][rank][key].reshape(-1);n+=a.numel()
  for start in range(0,a.numel(),1<<20):
   x=a[start:start+(1<<20)].float();y=b[start:start+(1<<20)].float();d=(x-y).double().abs();mx=max(mx,float(d.max()) if d.numel() else 0.);total+=float(d.sum());sq+=float(d.square().sum());unorm+=float(x.double().square().sum());rnorm+=float(y.double().square().sum());nz+=int(torch.count_nonzero(d))
 ck(key+' independent max mean L1 L2 norms count',n==row['numel'] and mx==row['max_abs'] and near(total,row['L1']) and near(total/n,row['mean_abs']) and near(math.sqrt(sq),row['L2']) and near(math.sqrt(unorm),row['U']['norm']) and near(math.sqrt(rnorm),row['R']['norm']) and (nz==0)==row['exact'] and nz/n==row['fraction_differing'])
 if index%50==0:print('FINAL_AUDIT_METRIC',index,flush=True)
bykey={x['key']:x for x in metrics['rows']}
for row in metrics['forward']:
 a=next(x for x in metas['U_LOC'][row['rank']]['forward_trace'] if x['module']==row['module'] and x['call']==row['call']);b=next(x for x in metas['R_LOC'][row['rank']]['forward_trace'] if x['module']==row['module'] and x['call']==row['call'])
 ck('forward hash comparison '+str(row['rank'])+row['module']+str(row['call']),row['different']==any(x['sha256']!=y['sha256'] for x,y in zip(a['outputs'],b['outputs'])))
for row in targets:
 o=row['observations'];early='PRE_EXISTING_PRE_FORWARD_STATE' if any(not v['exact'] for k,v in o.items() if k.startswith('pre_forward')) else 'FORWARD' if any(x['different'] for x in row['forward_observations']) else next((k for k in ['pre_reduction','post_reduction','pre_clip','post_clip','optimizer_exp_avg','optimizer_exp_avg_sq','post_step_parameter'] if not o[k]['exact']),'NO_OBSERVED_DIFFERENCE')
 ck('earliest stage '+row['target'],early==row['earliest_observable_pair_difference'])
 ck('target views agree with source metric rows '+row['target'],all(x==bykey[x['key']] for x in o.values()))
 expected='NO_DIVERGENCE_FOUND' if early=='NO_OBSERVED_DIFFERENCE' and all(x['available'] for x in o.values()) and row['forward_observations'] else 'TECHNICALLY_UNOBSERVABLE'
 ck('no causal label inferred from missing baseline '+row['target'],row['classification']==expected)
# Verify the actual frozen baseline's available stages, not an assumption.
baseline=[]
for i in range(1,6):
 folder=R/'runs/qwen3_4b_stage1_resume_v3/attempt2'/('U'+str(i));m=read(folder/'final_rank0.json');pf=read(folder/'preforward_step3_rank0.json')
 ck('original U'+str(i)+' final state is step5 only',m['progress']['update_step']==5 and all(k.startswith(('model.','optimizer.')) for k in m['tensors']))
 ck('original U'+str(i)+' preforward file records RNG and IDs, not gradient vectors','rng' in pf and 'tensors' not in pf)
 baseline.append({'run':'U'+str(i),'final_state_step':5,'preforward_step3_fields':sorted(pf)})
ck('original frozen CPU case preserved',source_cpu['statistical_result']['statistical_case']=='CASE_D' and not source_cpu['statistical_result']['confirmed_abnormal'])
ck('classification acknowledges genuine comparative evidence gap',summary['classification']=='INCONCLUSIVE' and summary['SAFE_TO_LAUNCH']=='NO' and all(x['exceeds_known_UU_natural_variation']=='UNDETERMINED_MISSING_MATCHED_STAGE_BASELINE' for x in targets))
ck('target counts recompute',summary['target_counts']==dict(collections.Counter(x['classification'] for x in targets)))
ck('cost recomputes',summary['new_gpu_optimizer_steps']==sum(x['optimizer_updates_observed'] for x in receipts) and summary['new_gpu_wall_seconds']==sum(x['wall_seconds'] for x in receipts))
ck('no V3 launch authority issued',not (R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V3.json').exists())
output={'status':'PASS','checks':checks,'check_count':len(checks),'baseline_availability_evidence':baseline,'scope':'Independent first-stage, state restoration, all localization tensor metrics, frozen evidence, read-only observer checks, and evidence-limited CASE_D closure. PASS means the audit is consistent; it does not mean numerical equivalence passes.','source_sha256':sha(Path(__file__)),'completed_at':time.time()}
(L/'INDEPENDENT_FINAL_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n');print('INDEPENDENT_FINAL_AUDIT_PASS',len(checks))
