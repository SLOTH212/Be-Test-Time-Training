import json,hashlib,itertools,math,statistics,time,sys
from pathlib import Path
import torch,numpy as np
from frozen_metrics import metrics
R=Path('/path/to/ttt');tag='qwen3_4b_stage1_resume_shared_boundary_v4_1';W=R/'work'/tag;A=R/'audits'/tag;RUN=R/'runs'/tag
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
P=A/'MISSING_CAPTURE_CLOSURE_PROTOCOL_V4_1.json';p=read(P);keys=p['promoted_keys'];names=p['branches'];receipts=read(A/'RUN_RECEIPTS.json');freeze=read(A/'PROTOCOL_FREEZE_RECEIPT.json')
assert [r['branch'] for r in receipts]==names and all(r['status']=='VALID' for r in receipts)
assert sum(r['optimizer_updates_observed'] for r in receipts)==6
assert all(r['protocol_sha256']==sha(P) and r['started_at']>freeze['frozen_at'] and r['observer_sha256']==p['observer_sha256'] for r in receipts)
for prev,nxt in zip(receipts,receipts[1:]):assert prev['ended_at']<nxt['started_at']
prior=read(A/'PRIOR_AUTHORITIES_IMMUTABLE.json')['files']
for path,h in prior.items():assert sha(path)==h,path
v4=read(Path(p['prior_v4_protocol_path']));assert p['decision']==v4['decision'] and p['metrics']==v4['metrics'] and p['scientific_contract']==v4['scientific_contract']
dcp=read(A/'SHARED_DCP_IDENTITY.json');root=Path(dcp['root']);latest=read(root/'latest.json');assert sha(root/'latest.json')==dcp['latest_sha256'] and sha(root/latest['slot']/'manifest.json')==dcp['manifest_sha256']
assert all(r['load_identity']==dcp for r in receipts[1:])
contracts=[];data={};meta={};training={};compact_files=[]
for n in names:
 data[n]=[];meta[n]=[]
 training[n]=[json.loads(line) for line in (RUN/n/'training.jsonl').read_text().splitlines()][-1]
 for rank in (0,1):
  out=RUN/n;saved=read(root/f'presave_step2_rank{rank}.json');actual=read(out/f'boundary_contract_rank{rank}.json')
  assert actual==saved==read(RUN/'A'/f'boundary_post_save_rank{rank}.json')
  assert read(RUN/'A'/f'boundary_save_rank{rank}.json')['status']=='PASS'
  assert read(out/f'initial_parent_check_rank{rank}.json')['status']=='PASS'
  assert saved['rank_state']['progress']['update_step']==2 and saved['rank_state']['progress']['record_cursor']==4 and saved['rank_state']['progress']['cumulative_tokens']==115086
  if n!='A':
   loaded=read(out/f'load_completeness_rank{rank}.json');assert loaded['status']=='PASS' and loaded['actual']==saved
  pre=read(out/f'preforward_step3_rank{rank}.json');ref=read(RUN/'A'/f'preforward_step3_rank{rank}.json');assert pre['rng']==ref['rng'] and pre['sample_ids']==ref['sample_ids'] and pre['cursor']==ref['cursor']==4 and pre['post_load_rng_preserved']
  m=read(out/f'final_rank{rank}.json');rawpath=out/f'final_rank{rank}.pt';assert sha(rawpath)==m['sha256'] and rawpath.stat().st_size==m['bytes'];d=torch.load(rawpath,map_location='cpu',weights_only=True)
  assert set(d)==set(m['tensors'])==set(keys) and len(d)==3
  assert m['runtime']['CUDA_VISIBLE_DEVICES']=='2,3' and m['runtime']['WORLD_SIZE']==2 and not m['runtime']['deterministic']
  assert m['progress']['update_step']==3 and m['progress']['record_cursor']==6
  for k in keys:
   info=m['tensors'][k];assert info['present'] and info['global_shape']==[2560] and info['local_shape']==[1280] and info['placement']=='(Shard(dim=0),)' and not info['replicated']
   assert list(d[k].shape)==[1280] and str(d[k].dtype)==info['dtype'] and torch.isfinite(d[k]).all()
  whole=read(out/f'whole_state_light_rank{rank}.json')
  def finite(v):
   if isinstance(v,dict):
    if 'finite_count' in v:assert v['finite_count']==v['numel']
    for value in v.values():finite(value)
   elif isinstance(v,list):
    for value in v:finite(value)
  finite(whole);compact_files.append({'path':str(out/f'whole_state_light_rank{rank}.json'),'sha256':sha(out/f'whole_state_light_rank{rank}.json')})
  co=m['clip_observation'];assert co['global_norm']['samples'][0]==training[n]['grad_norm_before_clip']
  contracts.append({'branch':n,'rank':rank,'shared_state_exact':True,'save_nonmutation':True,'rng_exact':True,'sample_ids':pre['sample_ids'],'cursor':pre['cursor'],'boundary_input_tokens':115086})
  data[n].append(d);meta[n].append(m)
 for field in ['update_step','record_cursor','cumulative_tokens','global_valid_tokens']:assert training[n][field]==training['A'][field]
 assert [r['sample_ids'] for r in training[n]['rank_runtime']]==[r['sample_ids'] for r in training['A']['rank_runtime']]
 for row in training[n]['rank_runtime']:assert row['physical_gpu'] in [2,3]
 assert meta[n][0]['clip_observation']==meta[n][1]['clip_observation']
files=[];full=A/'full_tensors';full.mkdir(exist_ok=True)
for n in names:
 for k in keys:
  short=k[len('post_step.optimizer.'):];path=full/(n+'.'+short+'.pt');tensor=torch.cat([data[n][r][k] for r in (0,1)],dim=0);assert list(tensor.shape)==[2560]
  torch.save(tensor,path);files.append({'branch':n,'key':k,'path':str(path),'sha256':sha(path),'shape':list(tensor.shape),'dtype':str(tensor.dtype),'numel':tensor.numel(),'source_shards':[{'path':str(RUN/n/f'final_rank{r}.pt'),'sha256':meta[n][r]['sha256'],'rank':r,'local_shape':[1280]} for r in (0,1)]})
pairs={}
for a,b in itertools.combinations(names,2):pairs[a+'_'+b]={k:metrics([(data[a][r][k],data[b][r][k]) for r in (0,1)],1e-6) for k in keys}
results={}
for k in keys:
 norms=[pairs['A_'+b][k]['right_norm'] for b in names[1:]];n=pairs['A_B1'][k]['numel'];ref=max(statistics.median(norms),1e-6*math.sqrt(n),np.finfo(np.float64).tiny)
 for pair,rows in pairs.items():rows[k]['relative_L2']=rows[k]['L2']/ref;rows[k]['common_reference_norm']=ref
 floors={'max_abs':1e-6,'mean_abs':1e-6,'L1':n*1e-6,'L2':math.sqrt(n)*1e-6,'relative_L2':math.sqrt(n)*1e-6/ref};calcs={}
 for metric,floor in floors.items():
  ab=[pairs['A_'+b][k][metric] for b in names[1:]];bb=[pairs[a+'_'+b][k][metric] for a,b in itertools.combinations(names[1:],2)];D=max(bb);N=min(ab);M=max(ab)
  calcs[metric]={'D':D,'N':N,'M':M,'floor':floor,'2D_plus_floor':2*D+floor,'D_plus_floor':D+floor,'max_condition_pass':M<=2*D+floor,'nearest_condition_pass':N<=D+floor,'consistent':M<=2*D+floor and N<=D+floor,'systematic_metric':N>2*D+floor}
 systematic=all(calcs[m]['systematic_metric'] for m in ['max_abs','L2']);consistent=all(c['consistent'] for c in calcs.values())
 result='SYSTEMATIC_A_VS_B_SEPARATION' if systematic else 'WITHIN_FRESH_LOAD_VARIABILITY' if consistent else 'MIXED_NOT_CONFIRMED'
 results[k]={'shape':[n],'common_reference_norm':ref,'calculations':calcs,'promotion_result':result,'all_pairs_exact':all(rows[k]['exact'] for rows in pairs.values())}
systematic_n=sum(r['promotion_result']=='SYSTEMATIC_A_VS_B_SEPARATION' for r in results.values());classification='PROCESS_RESTART_OR_RESUME_SPECIFIC_EFFECT' if systematic_n else 'SHARED_BOUNDARY_RESUME_CONSISTENT'
sanity={n:{'loss':training[n]['global_mean_loss'],'pre_clip_global_norm':training[n]['grad_norm_before_clip'],'actual_clip_coefficient':meta[n][0]['clip_observation']['actual_clamped_coefficient']['samples'][0],'update_step':training[n]['update_step'],'record_cursor':training[n]['record_cursor'],'cumulative_tokens':training[n]['cumulative_tokens'],'global_valid_tokens':training[n]['global_valid_tokens'],'step3_input_tokens':training[n]['cumulative_tokens']-115086,'record_ids':[r['sample_ids'] for r in training[n]['rank_runtime']],'finite':True} for n in names}
result={'schema':'V4_1_PROMOTED_FULL_TENSOR_COMPARISON','created_at':time.time(),'protocol_sha256':sha(P),'classification':classification,'systematic_promoted_state_n':systematic_n,'contracts':contracts,'full_tensor_files':files,'pairwise_metrics':pairs,'promoted_results':results,'sanity':sanity,'compact_fingerprints':compact_files,'shared_dcp_identity':dcp,'receipts':receipts,'new_gpu_optimizer_steps':6,'new_gpu_wall_seconds':sum(r['wall_seconds'] for r in receipts),'missing_mutable_state_n':0,'missing_promoted_capture_n':0,'prior_authorities_immutable':'PASS','original_48_evidence_reused':'PASS'}
write(A/'COMPARISON.json',result)
print(json.dumps({'classification':classification,'results':results,'sanity':sanity,'captures':len(files),'wall_seconds':result['new_gpu_wall_seconds']},indent=2))
