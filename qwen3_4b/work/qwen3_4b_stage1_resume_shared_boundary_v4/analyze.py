import json,hashlib,time,math,itertools,statistics,sys
from pathlib import Path
import torch,numpy as np
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';RUN=R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(8);assert not torch.cuda.is_initialized()
protocol=read(A/'SHARED_BOUNDARY_CAUSAL_PROTOCOL_V4.json');ph=sha(A/'SHARED_BOUNDARY_CAUSAL_PROTOCOL_V4.json');receipts=read(A/'RUN_RECEIPTS.json');names=[x['branch'] for x in receipts];assert names in [['A','B1','B2'],['A','B1','B2','B3']] and all(x['status']=='VALID' for x in receipts)
targets=read(A/'EXACT_V3_TARGETS.json')['rows'];parameters=sorted({x['parameter'] for x in targets});meta={n:[read(RUN/n/f'final_rank{r}.json') for r in (0,1)] for n in names}
contracts=[]
for rank in (0,1):
 saved=read(RUN/'A/checkpoints'/f'presave_step2_rank{rank}.json');assert saved==read(RUN/'A'/f'boundary_post_save_rank{rank}.json')
 assert saved['rank_state']['progress']['update_step']==2 and saved['rank_state']['progress']['record_cursor']==4 and saved['rank_state']['progress']['cumulative_tokens']==115086
 for n in names:
  actual=read(RUN/n/f'boundary_contract_rank{rank}.json');assert actual==saved,('BOUNDARY_MISMATCH',n,rank)
  if n!='A':
   load=read(RUN/n/f'load_completeness_rank{rank}.json');assert load['status']=='PASS' and load['actual']==saved
  assert read(RUN/n/f'initial_parent_check_rank{rank}.json')['status']=='PASS'
  info=read(RUN/n/f'preforward_step3_rank{rank}.json');ref=read(RUN/'A'/f'preforward_step3_rank{rank}.json');assert info['rng']==ref['rng'] and info['sample_ids']==ref['sample_ids'] and info['cursor']==ref['cursor']==4
  contracts.append({'branch':n,'rank':rank,'shared_state_exact':True,'RNG_exact':True,'sample_ids':info['sample_ids']})
tr={n:[json.loads(x) for x in (RUN/n/'training.jsonl').read_text().splitlines()][-1] for n in names}
for n,t in tr.items():
 for k in ['update_step','record_cursor','cumulative_tokens','global_valid_tokens']:assert t[k]==tr['A'][k]
 assert t['update_step']==3 and t['record_cursor']==6
 assert [x['sample_ids'] for x in t['rank_runtime']]==[x['sample_ids'] for x in tr['A']['rank_runtime']]
raw=[]
for n in names:
 for rank in (0,1):
  p=RUN/n/f'final_rank{rank}.pt';m=meta[n][rank];cache=A/(n+f'_RAW_HASH_rank{rank}.json')
  if cache.exists():
   c=read(cache);assert c['bytes']==p.stat().st_size and c['mtime_ns']==p.stat().st_mtime_ns and c['sha256']==m['sha256']
  else:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(16<<20),b''):h.update(b)
   assert h.hexdigest()==m['sha256'] and p.stat().st_size==m['bytes'];c={'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'verified_at':time.time()};cache.write_text(json.dumps(c,indent=2)+'\n')
  raw.append(c)
 print('RAW_HASH_OK',n,flush=True)
data={n:[torch.load(RUN/n/f'final_rank{r}.pt',mmap=True,map_location='cpu',weights_only=True) for r in (0,1)] for n in names}
keys=sorted(meta['A'][0]['tensors']);assert all(set(m['tensors'])==set(keys) for mm in meta.values() for m in mm)
def metrics(arrays,atol=None):
 n=0;mx=l1=sq=aa=bb=sa=sb=0.;nz=bad=closebad=0
 for a,b in arrays:
  assert a.shape==b.shape and a.dtype==b.dtype
  a=a.reshape(-1);b=b.reshape(-1)
  for off in range(0,a.numel(),1<<20):
   x=a[off:off+(1<<20)].float().numpy();y=b[off:off+(1<<20)].float().numpy();xx=x.astype(np.float64);yy=y.astype(np.float64);dd=np.abs(x-y).astype(np.float64)
   closebad+=int(np.count_nonzero(np.abs(x-y) > (atol+0.01*np.abs(y)))) if atol is not None else 0
   n+=len(x);bad+=int((~np.isfinite(x)).sum()+(~np.isfinite(y)).sum());nz+=int(np.count_nonzero(dd));mx=max(mx,float(dd.max(initial=0)));l1+=float(dd.sum());sq+=float(np.dot(dd,dd));aa+=float(np.dot(xx,xx));bb+=float(np.dot(yy,yy));sa+=float(xx.sum());sb+=float(yy.sum())
 assert not bad,'NONFINITE_CAPTURE'
 return {'numel':n,'exact':nz==0,'max_abs':mx,'mean_abs':l1/max(n,1),'L1':l1,'L2':math.sqrt(sq),'left_norm':math.sqrt(aa),'right_norm':math.sqrt(bb),'left_signed_mean':sa/max(n,1),'right_signed_mean':sb/max(n,1),'fraction_differing':nz/max(n,1),'finite':True,'legacy_allclose':closebad==0 if atol is not None else None,'legacy_mismatch_count':closebad if atol is not None else None}
pairs={}
for left,right in itertools.combinations(names,2):
 pair=left+'_'+right;cache=A/(pair+'_TARGET_METRICS.json')
 if cache.exists():
  obj=read(cache);assert obj['protocol_sha256']==ph;pairs[pair]=obj['rows'];continue
 rows={}
 for key in keys:
  masks=[[meta[n][r]['tensors'][key]['present'] for r in (0,1)] for n in [left,right]]
  assert all(all(m) for m in masks),('REQUIRED_CAPTURE_MISSING',pair,key)
  if key.startswith('post_reduce.'):
   alias=key.replace('post_reduce.','pre_clip.');continue
  if key.startswith('pre_forward.'):
   # Full raw shared-state equality established by exact boundary fingerprints.
   # Small step scalars still checked below; no redundant full before-state pair scan.
   continue
  rep=meta[left][0]['tensors'][key]['replicated'] and not key.startswith('pre_reduce.')
  ranks=[0] if rep else [0,1]
  if rep:
   for name in [left,right]:assert torch.equal(data[name][0][key],data[name][1][key])
  row=metrics([(data[left][r][key],data[right][r][key]) for r in ranks],(1e-5 if '.parameter.' in key else 1e-6) if key.startswith('post_step.') else None);rows[key]=row
  if len(rows)%40==0:print('METRICS',pair,len(rows),flush=True)
 for key in keys:
  if key.startswith('post_reduce.'):rows[key]=dict(rows[key.replace('post_reduce.','pre_clip.')],alias_of=key.replace('post_reduce.','pre_clip.'))
 obj={'DEBUG_ONLY':True,'protocol_sha256':ph,'source_sha256':sha(Path(__file__)),'pair':pair,'rows':rows};cache.write_text(json.dumps(obj,indent=2)+'\n');pairs[pair]=rows
print('PAIR_METRICS_COMPLETE',flush=True)
# Convert all compact observations into explicit fixed comparison features.
features={n:{} for n in names};feature_floor={};hash_patterns=[]
for n in names:
 for rank,m in enumerate(meta[n]):
  for event in m['forward_trace']:
   for j,out in enumerate(event['outputs']):
    assert out['finite_count']==out['numel'];prefix=f"forward.rank{rank}.{event['module']}.call{event['call']}.out{j}"
    features[n][prefix+'.samples']=out['samples'];feature_floor[prefix+'.samples']=1e-5
    for k in ['norm','mean','max_abs']:
     features[n][prefix+'.'+k]=[out[k]];feature_floor[prefix+'.'+k]=1e-5*(math.sqrt(out['numel']) if k=='norm' else 1)
  co=m['clip_observation'];assert co==meta[n][0]['clip_observation'];assert co['global_norm']['samples'][0]==tr[n]['grad_norm_before_clip']
 features[n]['loss']=[tr[n]['global_mean_loss']];feature_floor['loss']=1e-5
 for k in ['global_norm','actual_clamped_coefficient']:
  features[n]['clip.'+k]=meta[n][0]['clip_observation'][k]['samples'];feature_floor['clip.'+k]=1e-6
# Shapes/trace identities, all-element finite counts, and compact non-target checks.
for rank in (0,1):
 ref=meta['A'][rank]['forward_trace']
 for n in names:
  other=meta[n][rank]['forward_trace'];assert len(ref)==len(other)
  for x,y in zip(ref,other):
   assert (x['module'],x['call'],len(x['outputs']))==(y['module'],y['call'],len(y['outputs']))
   for a,b in zip(x['outputs'],y['outputs']):assert (a['shape'],a['dtype'],a['sample_indices'])==(b['shape'],b['dtype'],b['sample_indices'])
 for idx,event in enumerate(ref):
  for j,a in enumerate(event['outputs']):hash_patterns.append({'rank':rank,'module':event['module'],'call':event['call'],'output':j,'hashes':{n:meta[n][rank]['forward_trace'][idx]['outputs'][j]['sha256'] for n in names}})
 whole={n:read(RUN/n/f'whole_state_light_rank{rank}.json') for n in names}
 def flatten(v,prefix=''):
  if isinstance(v,dict) and 'sha256' in v and 'norm' in v:return {prefix:v}
  if isinstance(v,dict):
   out={}
   for k,z in v.items():out.update(flatten(z,prefix+'.'+k if prefix else k))
   return out
  return {}
 flats={n:flatten({k:v for k,v in whole[n].items() if k in ['model','optimizer','buffers']}) for n in names};assert all(set(f)==set(flats['A']) for f in flats.values())
 for key,f in flats['A'].items():
  for n in names:
   v=flats[n][key];assert (v['shape'],v['dtype'],v['global_shape'],v['placement'])==(f['shape'],f['dtype'],f['global_shape'],f['placement']);assert v['finite_count']==v['numel']
   if key.endswith('.step'):assert v['samples']==f['samples']
   for k in ['norm','mean','max_abs']:
    feature=f'whole.rank{rank}.{key}.{k}';features[n][feature]=[v[k]];base=1e-6 if key.startswith('optimizer.') else 1e-5;feature_floor[feature]=base*(math.sqrt(v['numel']) if k=='norm' else 1)
assert all(set(f)==set(features['A']) for f in features.values())
for pair in pairs:
 left,right=pair.split('_');rows=pairs[pair]
 for key,x in features[left].items():
  a=np.asarray(x,dtype=np.float64);b=np.asarray(features[right][key],dtype=np.float64);dd=np.abs(a-b);n=len(a)
  rows[key]={'numel':n,'exact':bool(np.array_equal(a,b)),'max_abs':float(dd.max(initial=0)),'mean_abs':float(dd.mean()) if n else 0.,'L1':float(dd.sum()),'L2':float(np.linalg.norm(dd)),'left_norm':float(np.linalg.norm(a)),'right_norm':float(np.linalg.norm(b)),'finite':bool(np.isfinite(a).all() and np.isfinite(b).all())}
# Same floor/diameter/nearest rules for every measured target stage and compact feature.
fresh=names[1:];ab=['A_'+n for n in fresh];bb=[x+'_'+y for x,y in itertools.combinations(fresh,2)];decisions=[]
for key in pairs[ab[0]]:
 rows={p:pairs[p][key] for p in pairs};n=rows[ab[0]]['numel']
 atol=feature_floor.get(key,1e-5 if key.startswith('post_step.parameter.') else 1e-6)
 norms=[pairs['A_'+b][key]['right_norm'] for b in fresh];scale=max(statistics.median(norms),atol*math.sqrt(n),np.finfo(float).tiny)
 floors={'max_abs':atol,'mean_abs':atol,'L1':n*atol,'L2':math.sqrt(n)*atol,'relative_L2':math.sqrt(n)*atol/scale}
 for row in rows.values():row['relative_L2']=row['L2']/scale
 detail={}
 for metric,floor in floors.items():
  diameter=max(rows[p][metric] for p in bb);nearest=min(rows[p][metric] for p in ab);maximum=max(rows[p][metric] for p in ab)
  detail[metric]={'fresh_diameter':diameter,'nearest_A':nearest,'maximum_A':maximum,'floor':floor,'consistent':maximum<=2*diameter+floor and nearest<=diameter+floor,'all_A_separated':nearest>2*diameter+floor}
 decisions.append({'key':key,'consistent':all(x['consistent'] for x in detail.values()),'systematic_candidate':detail['max_abs']['all_A_separated'] and detail['L2']['all_A_separated'],'metrics':detail})
failed=[x for x in decisions if not x['consistent']];systematic=[x for x in decisions if x['systematic_candidate']]
operator=read(A/'OPERATOR_AUDIT.json');assert operator['all_reductions_match_fp32_two_rank_average'] and operator['all_clips_match_actual_recorded_coefficient']
assert set(x['run'] for x in operator['results'])==set(names)
run_b3=bool(failed) and len(names)==3
cls='SHARED_BOUNDARY_RESUME_CONSISTENT' if not failed else 'PENDING_PREREGISTERED_B3' if run_b3 else 'PROCESS_RESTART_OR_RESUME_SPECIFIC_EFFECT' if systematic else 'INCONCLUSIVE'
result={'DEBUG_ONLY':True,'protocol_sha256':ph,'classification':cls,'run_B3':run_b3,'branches':names,'contracts':contracts,'shared_boundary_identity':'PASS','save_nonmutation':'PASS','RNG_parity':'PASS','target_n':48,'parameter_n':47,'failed_observable_n':len(failed),'systematic_candidate_n':len(systematic),'failed_observables':failed,'systematic_candidates':systematic,'all_observable_decisions':decisions,'forward_hash_patterns':hash_patterns,'raw_files':raw,'training_step3':tr,'new_gpu_optimizer_steps':sum(x['optimizer_updates_observed'] for x in receipts),'new_gpu_wall_seconds':sum(x['wall_seconds'] for x in receipts),'source_sha256':sha(Path(__file__)),'completed_at':time.time()}
(A/('ABC_DECISION.json' if len(names)==3 else 'ABCD_DECISION.json')).write_text(json.dumps(result,indent=2)+'\n');(A/'ALL_COMPARISON_METRICS.json').write_text(json.dumps({'DEBUG_ONLY':True,'protocol_sha256':ph,'pairs':pairs},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['classification','run_B3','failed_observable_n','systematic_candidate_n','new_gpu_optimizer_steps','new_gpu_wall_seconds']},indent=2),flush=True)
