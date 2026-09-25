import json,math,hashlib,time,collections
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/ttt');A=R/'audits/v3';L=A/'localization_final';RUN=R/'runs/qwen3_4b_stage1_resume_v3/localization_final'
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(L/'CONDITIONAL_LOCALIZATION_PLAN.json');targets=plan['targets'];receipts=read(L/'LOCALIZATION_RUN_RECEIPTS.json');assert len(receipts)==3 and all(x['status']=='VALID_COMPLETED_PHASE' for x in receipts)
meta={n:[read(RUN/n/f'final_rank{k}.json') for k in (0,1)] for n in ['U_LOC','R_LOC']};data={n:[torch.load(RUN/n/f'final_rank{k}.pt',map_location='cpu',mmap=True,weights_only=True) for k in (0,1)] for n in meta}
raw_files=[]
for n in meta:
 for rank in (0,1):
  p=RUN/n/f'final_rank{rank}.pt';h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(16<<20),b''):h.update(b)
  assert h.hexdigest()==meta[n][rank]['sha256'] and p.stat().st_size==meta[n][rank]['bytes'];raw_files.append({'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size})
  assert meta[n][rank]['observer_read_checks']>0
print('RAW_LOCALIZATION_FILES_VERIFIED',flush=True)
contracts=[]
for rank in (0,1):
 u=read(RUN/'U_LOC'/f'boundary_contract_rank{rank}.json');r=read(RUN/'R_LOC_pre/checkpoints'/f'presave_step2_rank{rank}.json');loaded=read(RUN/'R_LOC'/f'boundary_contract_rank{rank}.json');postload=read(RUN/'R_LOC'/f'load_completeness_rank{rank}.json')
 assert loaded==r and postload['status']=='PASS' and postload['actual']==r
 assert u['rank_state']==r['rank_state'] and u['param_groups']==r['param_groups'] and u['runtime']==r['runtime'] and u['buffers']==r['buffers']
 for name in ['U_LOC','R_LOC_pre','R_LOC']:assert read(RUN/name/f'initial_parent_check_rank{rank}.json')['status']=='PASS'
 contracts.append({'rank':rank,'state_after_load_exact':True,'before_forward_exact_to_saved':True,'input_rng_scheduler_cursor_groups_exact':True,'U_R_preforward_model_hash_differences':sum(u['model'][k]!=r['model'][k] for k in u['model']),'U_R_preforward_optimizer_parameter_hash_differences':sum(u['optimizer'][k]!=r['optimizer'][k] for k in u['optimizer'])})
trajectories={n:[json.loads(x) for x in (RUN/n/'training.jsonl').read_text().splitlines()] for n in ['U_LOC','R_LOC_pre','R_LOC']}
u=trajectories['U_LOC'];r=trajectories['R_LOC_pre']+trajectories['R_LOC'];assert len(u)==len(r)==3
for x,y in zip(u,r):
 for key in ['update_step','record_cursor','cumulative_tokens']:assert x[key]==y[key]
 assert [v['sample_ids'] for v in x['rank_runtime']]==[v['sample_ids'] for v in y['rank_runtime']]
old=read(A/'REPLICATE_INPUT_PARITY.json')['trajectories'];scalar_baseline={}
for step in (1,2,3):
 scalar_baseline[str(step)]={}
 for key,oldkey in [('loss','loss'),('gradient_norm_before_clip','gradient_norm_before_clip')]:
  vals=[old['U'+str(i)][step-1][oldkey] for i in range(1,6)];uu=max(vals)-min(vals);newkey='global_mean_loss' if key=='loss' else 'grad_norm_before_clip';diff=abs(u[step-1][newkey]-r[step-1][newkey]);scalar_baseline[str(step)][key]={'original5_U_values':vals,'original_UU_max_abs':uu,'U_LOC':u[step-1][newkey],'R_LOC':r[step-1][newkey],'local_pair_abs_difference':diff,'within_observed_UU_range':diff<=uu}
keys=sorted(meta['U_LOC'][0]['tensors']);assert all(set(m['tensors'])==set(keys) for rows in meta.values() for m in rows)
rows=[]
for key in keys:
 masks={n:[meta[n][k]['tensors'][key]['present'] for k in (0,1)] for n in meta}
 if not all(all(x) for x in masks.values()):
  rows.append({'key':key,'available':False,'presence':masks,'exact':masks['U_LOC']==masks['R_LOC'],'reason':'Absent or not safely observed gradient/state; not treated as zero'});continue
 pre_reduce=key.startswith('pre_reduce.')
 replicated=meta['U_LOC'][0]['tensors'][key]['replicated'] and not pre_reduce
 ranks=[0] if replicated else [0,1]
 if replicated:
  for name in data:assert torch.equal(data[name][0][key],data[name][1][key])
 n=sum(data['U_LOC'][rank][key].numel() for rank in ranks);mx=l1=sq=unorm=rnorm=usigned=rsigned=umax=rmax=uabs=rabs=legacy_sq=0.;nz=0;finite=True
 for rank in ranks:
  left=data['U_LOC'][rank][key].reshape(-1);right=data['R_LOC'][rank][key].reshape(-1);assert left.shape==right.shape and left.dtype==right.dtype
  for start in range(0,left.numel(),1<<20):
   a=left[start:start+(1<<20)].float().numpy();b=right[start:start+(1<<20)].float().numpy();ad=a.astype(np.float64);bd=b.astype(np.float64);diff=np.abs(a-b);dd=diff.astype(np.float64)
   finite=finite and bool(np.isfinite(a).all() and np.isfinite(b).all());mx=max(mx,float(diff.max(initial=0)));l1+=float(dd.sum());sq+=float(np.dot(dd,dd));nz+=int(np.count_nonzero(diff));legacy_sq+=float(np.square(diff).sum(dtype=np.float64));unorm+=float(np.dot(ad,ad));rnorm+=float(np.dot(bd,bd));usigned+=float(ad.sum());rsigned+=float(bd.sum());uabs+=float(np.abs(ad).sum());rabs+=float(np.abs(bd).sum());umax=max(umax,float(np.abs(a).max(initial=0)));rmax=max(rmax,float(np.abs(b).max(initial=0)))
  
 assert finite,('NONFINITE_LOCALIZATION_VALUE',key)
 l2=math.sqrt(sq);row={'key':key,'available':True,'numel':n,'max_abs':mx,'mean_abs':l1/n,'L1':l1,'L2':l2,'pair_reference_relative_L2':l2/max(math.sqrt(unorm),np.finfo(float).tiny),'legacy_rms_float32_square':math.sqrt(legacy_sq/n),'exact':nz==0,'fraction_differing':nz/n,'finite':finite,'U':{'norm':math.sqrt(unorm),'max_abs':umax,'mean_abs':uabs/n,'signed_mean':usigned/n},'R':{'norm':math.sqrt(rnorm),'max_abs':rmax,'mean_abs':rabs/n,'signed_mean':rsigned/n},'layout':'concatenated rank-local unsharded pre-reduction contributions' if pre_reduce else 'replicated scalar counted once' if replicated else 'FSDP shards concatenated','matched_original5_U_baseline':'NOT_AVAILABLE_AT_THIS_STAGE'};rows.append(row)
 if len(rows)%25==0:print('LOCALIZATION_METRIC',len(rows),len(keys),flush=True)
bykey={x['key']:x for x in rows};forward=[]
for rank in (0,1):
 ut=meta['U_LOC'][rank]['forward_trace'];rt=meta['R_LOC'][rank]['forward_trace'];assert [(x['module'],x['call']) for x in ut]==[(x['module'],x['call']) for x in rt]
 for x,y in zip(ut,rt):
  assert len(x['outputs'])==len(y['outputs']);different=False;outs=[]
  for a,b in zip(x['outputs'],y['outputs']):
   assert a['finite_count']==a['numel'] and b['finite_count']==b['numel'];assert a['shape']==b['shape'] and a['dtype']==b['dtype'] and a['sample_indices']==b['sample_indices'];different|=a['sha256']!=b['sha256'];outs.append({'U':a,'R':b,'exact_hash_match':a['sha256']==b['sha256'],'sample_max_abs_diff':max([abs(v-w) for v,w in zip(a['samples'],b['samples'])] or [0.])})
  forward.append({'rank':rank,'module':x['module'],'call':x['call'],'ordinal':x['ordinal'],'different':different,'outputs':outs,'matched_original5_U_baseline':'NOT_AVAILABLE'})
clip={n:[m['clip_observation'] for m in mm] for n,mm in meta.items()}
for n in clip:
 assert clip[n][0]==clip[n][1]
 assert clip[n][0]['actual_clamped_coefficient']['numel']==1
 assert clip[n][0]['global_norm']['samples'][0]==trajectories[n][-1]['grad_norm_before_clip']
results=[]
for target in targets:
 name=target['parameter'];module=target['module'];state=target['state_type'];prekey='pre_forward.'+('parameter.'+name if state=='parameter' else 'optimizer.'+name+'.'+state);postkey='post_step.'+('parameter.'+name if state=='parameter' else 'optimizer.'+name+'.'+state)
 stagekeys={'pre_forward':prekey,'pre_forward_parameter':'pre_forward.parameter.'+name,'pre_forward_exp_avg':'pre_forward.optimizer.'+name+'.exp_avg','pre_forward_exp_avg_sq':'pre_forward.optimizer.'+name+'.exp_avg_sq','pre_forward_step':'pre_forward.optimizer.'+name+'.step','pre_reduction':'pre_reduce.gradient.'+name,'post_reduction':'post_reduce.gradient.'+name,'pre_clip':'pre_clip.gradient.'+name,'post_clip':'post_clip.gradient.'+name,'optimizer_exp_avg':'post_step.optimizer.'+name+'.exp_avg','optimizer_exp_avg_sq':'post_step.optimizer.'+name+'.exp_avg_sq','optimizer_step':'post_step.optimizer.'+name+'.step','post_step_parameter':'post_step.parameter.'+name,'target_state_post_step':postkey}
 observed={stage:bykey[key] for stage,key in stagekeys.items()};ft=[x for x in forward if x['module']==module];earliest='PRE_EXISTING_PRE_FORWARD_STATE' if any(not v['exact'] for k,v in observed.items() if k.startswith('pre_forward')) else 'FORWARD' if any(x['different'] for x in ft) else next((stage for stage in ['pre_reduction','post_reduction','pre_clip','post_clip','optimizer_exp_avg','optimizer_exp_avg_sq','post_step_parameter'] if not observed[stage]['exact']),'NO_OBSERVED_DIFFERENCE')
 # A final step5 state distribution is not a step3 activation/gradient baseline.
 # Classifier does not turn missing comparative evidence into natural variation.
 cls='NO_DIVERGENCE_FOUND' if earliest=='NO_OBSERVED_DIFFERENCE' and all(v['available'] for v in observed.values()) and ft else 'TECHNICALLY_UNOBSERVABLE'
 results.append({'target':target['key'],'parameter':name,'module':module,'shape':target['shape'],'dtype':target['dtype'],'cpu_trigger_metrics':[k for k,v in target['cpu_trigger']['metrics'].items() if not v['extreme_pass']],'postload_vs_saved_exact':True,'observations':observed,'forward_observations':ft,'earliest_observable_pair_difference':earliest,'exceeds_known_UU_natural_variation':'UNDETERMINED_MISSING_MATCHED_STAGE_BASELINE','classification':cls,'classification_scope':'TECHNICALLY_UNOBSERVABLE refers to the requested resume-specific vs natural-variation discriminator, not a failure to capture the numeric values. No step5 baseline substituted for step3 data.'})
summary={'status':'LOCALIZATION_CAPTURE_COMPLETE','classification':'INCONCLUSIVE','statistical_case':'CASE_D','reason':'Localization pair captured actual stages and verified exact reload, but original5 U baseline lacks matching first-post-boundary activation/gradient/actual-clipping/step3-moment vectors. Single U_LOC/R_LOC pair cannot distinguish pre-existing original-mode variation from resume-specific excess for these observables. Original frozen empirical gates remain inconclusive, not relaxed.','target_counts':dict(collections.Counter(x['classification'] for x in results)),'target_n':48,'parameter_n':47,'postload_preforward_exact':True,'pre_boundary_contract':'PASS','contracts':contracts,'scalar_baseline':scalar_baseline,'clip':clip,'earliest_pair_difference_counts':dict(collections.Counter(x['earliest_observable_pair_difference'] for x in results)),'resume_specific_target_n':0,'resume_specific_count_meaning':'No target confirmed; this does not prove absence.','root_cause':'MIXED_OR_UNRESOLVED_ORIGINAL_MODE_VARIATION_VS_RESUME_EFFECT','clip_path_resume_specific':'NOT_DEMONSTRATED','new_gpu_optimizer_steps':sum(x['optimizer_updates_observed'] for x in receipts),'new_gpu_wall_seconds':sum(x['wall_seconds'] for x in receipts),'SAFE_TO_LAUNCH':'NO','completed_at':time.time()}
(L/'LOCALIZATION_METRICS.json').write_text(json.dumps({'validated_metric_source_sha256':sha(R/'work/qwen3_4b_stage1_resume_v3/v3_metrics_cached_tolerance.py'),'rows':rows,'forward':forward,'raw_files':raw_files},indent=2)+'\n');(L/'PER_TARGET_LOCALIZATION.json').write_text(json.dumps(results,indent=2)+'\n');(L/'LOCALIZATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
