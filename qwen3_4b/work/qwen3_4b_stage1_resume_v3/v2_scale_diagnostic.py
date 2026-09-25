import json,math,collections,hashlib
from pathlib import Path
import numpy as np,torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
R=Path('/path/to/ttt');A=R/'audits/v3';torch.set_num_threads(8)
inv=json.loads((A/'V2_FAILED_76_TENSOR_INVENTORY.json').read_text())
def flat(d,p=''):
 out={}
 if isinstance(d,dict):
  for k,v in d.items():out.update(flat(v,p+'.'+str(k)))
 elif isinstance(d,(list,tuple)):
  for k,v in enumerate(d):out.update(flat(v,p+'.'+str(k)))
 elif isinstance(d,torch.Tensor):out[p]=d
 return out
states=[];paths=['U1/checkpoints/slot_B','U2/checkpoints/slot_B','R_checkpoints/slot_B']
for rel in paths:
 p=R/'runs/qwen3_4b_stage1_closure_v2'/rel;d={};_load_state_dict(d,storage_reader=FileSystemReader(p/'dcp'),planner=_EmptyStateDictLoadPlanner(),no_dist=True);states.append({g:flat(d[g]) for g in ['model','optimizer']});print('loaded',rel,flush=True)
def qhist(h,qs,n):
 vv=np.array(sorted(h));cc=np.cumsum([h[float(v)] for v in vv],dtype=np.int64);out={}
 for q in qs:
  pos=q*(n-1);lo=int(math.floor(pos));hi=int(math.ceil(pos));a=vv[np.searchsorted(cc,lo,side='right')];b=vv[np.searchsorted(cc,hi,side='right')];out[str(q)]=float(a+(b-a)*(pos-lo))
 return out
for z,row in enumerate(inv['rows']):
 vals=[s[row['group']][row['canonical_state_name']].reshape(-1) for s in states];n=vals[0].numel();atol=row['atol'];rtol=row['rtol'];scale=[{'sum_abs':0.,'sum_sq':0.,'sum_signed':0.,'max_abs':0.} for _ in vals];pairs=[(0,1,'U1_U2'),(0,2,'U1_R'),(1,2,'U2_R')];stats={p:{'differing_n':0,'old_allclose_violation_n':0,'near_zero_reference_n':0,'near_zero_violation_n':0,'absolute_term_dominant_violation_n':0,'hist':collections.Counter()} for _,_,p in pairs}
 for start in range(0,n,1<<20):
  vv=[x[start:start+(1<<20)].float() for x in vals]
  for i,x in enumerate(vv):
   s=scale[i];s['sum_abs']+=float(x.abs().sum(dtype=torch.float64));s['sum_sq']+=float(x.square().sum(dtype=torch.float64));s['sum_signed']+=float(x.sum(dtype=torch.float64));s['max_abs']=max(s['max_abs'],float(x.abs().max()))
  for i,j,p in pairs:
   x,y=vv[i],vv[j];d=(x-y).abs();v=~torch.isclose(x,y,rtol=rtol,atol=atol);near=x.abs()<=atol/rtol;s=stats[p];s['differing_n']+=int(torch.count_nonzero(d));s['old_allclose_violation_n']+=int(v.sum());s['near_zero_reference_n']+=int(near.sum());s['near_zero_violation_n']+=int((near&v).sum());s['absolute_term_dominant_violation_n']+=int((v&(atol>=rtol*x.abs())).sum());values,counts=np.unique(d.numpy(),return_counts=True);s['hist'].update({float(k):int(v) for k,v in zip(values,counts)})
 row['value_scales']={name:{'tensor_norm':math.sqrt(s['sum_sq']),'max_abs_value':s['max_abs'],'mean_abs_value':s['sum_abs']/n,'RMS_value':math.sqrt(s['sum_sq']/n),'signed_mean':s['sum_signed']/n} for name,s in zip(['U1','U2','R'],scale)}
 for p,s in stats.items():
  s['absolute_difference_quantiles']=qhist(s.pop('hist'),[.5,.95,.99,1.],n);s['fraction_elements_differing']=s['differing_n']/n;s['fraction_violating_old_allclose']=s['old_allclose_violation_n']/n;s['fraction_violations_near_zero']=s['near_zero_violation_n']/max(1,s['old_allclose_violation_n']);s['support_class']='NO_OLD_ALLCLOSE_VIOLATIONS' if not s['old_allclose_violation_n'] else ('SPARSE_ISOLATED' if s['fraction_violating_old_allclose']<=1e-4 else ('BROAD' if s['fraction_violating_old_allclose']>=.01 else 'INTERMEDIATE_SUPPORT'));s['near_zero_dominant']=s['fraction_violations_near_zero']>=.5
 row['scale_diagnostics']=stats
 print('tensor',z+1,row['canonical_state_name'],flush=True)
inv['status']='PASS';inv['FAILED_TENSOR_SCALE_DIAGNOSTIC']='PASS';inv['quantile_method']='Exact full-element float32 absolute-difference frequency histograms; linear order-statistic interpolation including zeros. No subsampling.'
(A/'V2_FAILED_76_TENSOR_INVENTORY.json').write_text(json.dumps(inv,indent=2)+'\n')
out={'status':'PASS','FAILED_TENSOR_SCALE_DIAGNOSTIC':'PASS','diagnostic_only_no_threshold_changes':True,'tensor_n':76,'rows':[{k:x[k] for k in ['group','canonical_state_name','parameter','state_type','value_scales','scale_diagnostics','criterion_failure']} for x in inv['rows']]}
(A/'FAILED_TENSOR_SCALE_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n');print('SCALE_DIAGNOSTIC_PASS',flush=True)
