import json,sys,math,time
from pathlib import Path
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
R=Path('/path/to/ttt');A=R/'audits';RUN=R/'runs/qwen3_4b_stage1_closure_v2';torch.set_num_threads(8)
paths=[RUN/'U1/checkpoints/slot_B',RUN/'U2/checkpoints/slot_B',RUN/'R_checkpoints/slot_B']
states=[]
for p in paths:
 m=json.loads((p/'manifest.json').read_text());assert m['complete'] and m['progress']['update_step']==5
 d={};_load_state_dict(d,storage_reader=FileSystemReader(p/'dcp'),planner=_EmptyStateDictLoadPlanner(),no_dist=True);states.append(d)
def flat(d,prefix=''):
 out={}
 if isinstance(d,dict):
  for k,v in d.items():out.update(flat(v,prefix+'.'+str(k)))
 elif isinstance(d,(list,tuple)):
  for k,v in enumerate(d):out.update(flat(v,prefix+'.'+str(k)))
 elif isinstance(d,torch.Tensor):out[prefix]=d
 return out
pairs=[(0,1,'U1_U2'),(0,2,'U1_R'),(1,2,'U2_R')]
report={};all_abnormal=[]
for group,atol in [('model',1e-5),('optimizer',1e-6)]:
 tensors=[flat(d[group]) for d in states];assert tensors[0].keys()==tensors[1].keys()==tensors[2].keys()
 rows=[];aggregate={name:{'numel':0,'tensor_count':0,'exact_tensors':0,'old_allclose_tensors':0,'max_abs':0.,'sum_abs':0.,'sum_square_diff':0.,'sum_square_reference':0.,'sum_square_other':0.} for _,_,name in pairs}
 for key in tensors[0]:
  vals=[d[key] for d in tensors];assert all(v.shape==vals[0].shape and v.dtype==vals[0].dtype for v in vals)
  metric={name:{'max_abs':0.,'sum_abs':0.,'sum_square_diff':0.,'sum_square_reference':0.,'sum_square_other':0.,'exact':True,'old_allclose':True} for _,_,name in pairs}
  for start in range(0,vals[0].numel(),1<<20):
   vv=[v.reshape(-1)[start:start+(1<<20)].float() for v in vals]
   for i,j,name in pairs:
    u,v=vv[i],vv[j];z=(u-v).abs();m=metric[name]
    m['max_abs']=max(m['max_abs'],float(z.max()) if z.numel() else 0.);m['sum_abs']+=float(z.sum(dtype=torch.float64));m['sum_square_diff']+=float(z.square().sum(dtype=torch.float64));m['sum_square_reference']+=float(u.square().sum(dtype=torch.float64));m['sum_square_other']+=float(v.square().sum(dtype=torch.float64))
    m['exact']=m['exact'] and torch.equal(u,v);m['old_allclose']=m['old_allclose'] and bool(torch.isclose(u,v,rtol=.01,atol=atol).all())
  n=vals[0].numel()
  for _,_,name in pairs:
   m=metric[name];g=aggregate[name];g['numel']+=n;g['tensor_count']+=1;g['exact_tensors']+=int(m['exact']);g['old_allclose_tensors']+=int(m['old_allclose']);g['max_abs']=max(g['max_abs'],m['max_abs'])
   for k in ['sum_abs','sum_square_diff','sum_square_reference','sum_square_other']:g[k]+=m[k]
   m.update(mean_abs=m['sum_abs']/max(1,n),rms_diff=math.sqrt(m['sum_square_diff']/max(1,n)),l2_diff=math.sqrt(m['sum_square_diff']),relative_norm=math.sqrt(m['sum_square_diff']/max(1e-300,m['sum_square_reference'])),parameter_norm_difference=abs(math.sqrt(m['sum_square_reference'])-math.sqrt(m['sum_square_other'])))
  base=metric['U1_U2'];ok=all(metric[k]['old_allclose'] for k in ['U1_R','U2_R']) or all(metric[k]['max_abs']<=base['max_abs']*1.01+atol and metric[k]['rms_diff']<=base['rms_diff']*1.01+atol for k in ['U1_R','U2_R'])
  if not ok:all_abnormal.append({'group':group,'key':key,'metrics':metric})
  rows.append({'key':key,'shape':list(vals[0].shape),'dtype':str(vals[0].dtype),'metrics':metric,'within_predeclared_empirical_envelope_or_old_tolerance':ok})
 for name,g in aggregate.items():g.update(mean_abs=g['sum_abs']/max(1,g['numel']),l2_diff=math.sqrt(g['sum_square_diff']),relative_norm=math.sqrt(g['sum_square_diff']/max(1e-300,g['sum_square_reference'])),parameter_norm_difference=abs(math.sqrt(g['sum_square_reference'])-math.sqrt(g['sum_square_other'])))
 report[group]={'aggregate':aggregate,'tensors':rows,'old_tolerance':{'rtol':.01,'atol':atol}}
u1=list(map(json.loads,(RUN/'U1/training.jsonl').read_text().splitlines()));u2=list(map(json.loads,(RUN/'U2/training.jsonl').read_text().splitlines()));rr=list(map(json.loads,(RUN/'R_recovered/training.jsonl').read_text().splitlines()));res=u1[:2]+rr
assert [x['update_step'] for x in res]==[1,2,3,4,5]
def ids(rows):return [[rank['sample_ids'] for rank in row['rank_runtime']] for row in rows]
assert ids(u1)==ids(u2)==ids(res)
loss=[]
for x,y,z in zip(u1,u2,res):
 values=[q['global_mean_loss'] for q in [x,y,z]];d12=abs(values[0]-values[1]);d1r=abs(values[0]-values[2]);d2r=abs(values[1]-values[2]);old_ok=all(d<=1e-5+.01*abs(values[i]) for i,d in [(0,d1r),(1,d2r)]);ok=old_ok or max(d1r,d2r)<=d12*1.01+1e-5;loss.append({'step':x['update_step'],'values_U1_U2_R':values,'U1_U2_abs':d12,'U1_R_abs':d1r,'U2_R_abs':d2r,'accepted':ok})
init=[]
for rank in [0,1]:
 a=json.loads((RUN/f'U1/initial_rank{rank}.json').read_text());b=json.loads((RUN/f'U2/initial_rank{rank}.json').read_text());init.append({'rank':rank,'initial_model_optimizer_groups_buffers_rng_exact':a==b})
assert all(x['initial_model_optimizer_groups_buffers_rng_exact'] for x in init)
passed=not all_abnormal and all(x['accepted'] for x in loss)
out={'status':'PASS_NUMERICALLY_COMPARABLE' if passed else 'FAIL_ABNORMAL_OR_INSUFFICIENT_BASELINE','ORIGINAL_MODE_RESUME_NUMERICAL_PARITY':'WARNING_EXPECTED_NONDETERMINISM' if passed else 'FAIL','plan':str(A/'ORIGINAL_MODE_BASELINE_COMPARISON_PLAN_V2.json'),'initial_state':init,'canonical_sequence_exact':True,'loss_per_step':loss,'comparisons':report,'abnormal_tensor_count':len(all_abnormal),'abnormal_tensors':all_abnormal,'deterministic_mode_enabled':False,'statistical_scope':'Empirical numerical envelope from two independent uninterrupted runs, not a population significance claim.'}
(A/'ORIGINAL_MODE_NONDETERMINISM_BASELINE_V2.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'abnormal_tensor_count':len(all_abnormal),'model':report['model']['aggregate'],'optimizer':report['optimizer']['aggregate'],'loss':loss},indent=2),flush=True)
