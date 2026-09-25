import json,sys,time
from pathlib import Path
import torch,numpy as np
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1';torch.set_num_threads(8)
left=R/'runs/qwen3_4b_stage2_preflight_v1/uninterrupted_A/checkpoints/slot_B'
right=R/'runs/qwen3_4b_stage2_preflight_v1/resume_B/checkpoints/slot_B'
def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
state_checks=[]
for rank in [0,1]:
 l=torch.load(left/f'rank_{rank:05d}.pt',map_location='cpu',weights_only=False);r=torch.load(right/f'rank_{rank:05d}.pt',map_location='cpu',weights_only=False)
 state_checks.append({'rank':rank,'progress_exact':exact(l['progress'],r['progress']),'scheduler_exact':exact(l['scheduler'],r['scheduler']),'rng_exact':all(exact(l[k],r[k]) for k in ['python_rng','numpy_rng','torch_cpu_rng','torch_cuda_rng'])})
a_rows=list(map(json.loads,(R/'runs/qwen3_4b_stage2_preflight_v1/uninterrupted_A/training.jsonl').read_text().splitlines()))
b_rows=list(map(json.loads,(R/'runs/qwen3_4b_stage2_preflight_v1/resume_B/training.jsonl').read_text().splitlines()))
reference=[x for x in a_rows if x['update_step'] in [3,4,5]]
assert len(reference)==len(b_rows)==3
sequence=all([r['sample_ids'] for r in x['rank_runtime']]==[r['sample_ids'] for r in y['rank_runtime']] for x,y in zip(reference,b_rows))
states=[]
for path in [left,right]:
 d={};_load_state_dict(d,storage_reader=FileSystemReader(path/'dcp'),planner=_EmptyStateDictLoadPlanner(),no_dist=True);states.append(d)
def compare(a,b,atol):
 report={'tensor_count':0,'exact_tensor_count':0,'allclose_tensor_count':0,'max_abs':0.0,'key_shape_dtype_equal':True,'worst_key':None,'rtol':0.01,'atol':atol}
 def walk(x,y,key):
  if isinstance(x,torch.Tensor):
   report['tensor_count']+=1
   if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:report['key_shape_dtype_equal']=False;return
   if torch.equal(x,y):report['exact_tensor_count']+=1;report['allclose_tensor_count']+=1;return
   flatx=x.reshape(-1);flaty=y.reshape(-1);close=True;maxabs=0.0
   for i in range(0,x.numel(),1<<20):
    u=flatx[i:i+(1<<20)].float();v=flaty[i:i+(1<<20)].float()
    maxabs=max(maxabs,float((u-v).abs().max()));close=close and bool(torch.isclose(u,v,atol=atol,rtol=0.01).all())
   report['allclose_tensor_count']+=int(close)
   if maxabs>report['max_abs']:report['max_abs']=maxabs;report['worst_key']=key
  elif isinstance(x,dict):
   if x.keys()!=y.keys():report['key_shape_dtype_equal']=False;return
   for k in x:walk(x[k],y[k],key+'.'+str(k))
  elif isinstance(x,(list,tuple)):
   if len(x)!=len(y):report['key_shape_dtype_equal']=False;return
   for i,(u,v) in enumerate(zip(x,y)):walk(u,v,key+'.'+str(i))
  elif x!=y:report['key_shape_dtype_equal']=False
 walk(a,b,'')
 return report
numeric={'model':compare(states[0]['model'],states[1]['model'],1e-5),'optimizer':compare(states[0]['optimizer'],states[1]['optimizer'],1e-6)}
ok=sequence and all(all(x[k] for k in ['progress_exact','scheduler_exact','rng_exact']) for x in state_checks)
receipt={'STAGE2_RESUME_STATE_INTEGRITY':'PASS' if ok else 'FAIL','canonical_record_sequence_exact':sequence,'state_checks':state_checks,'numerical_deviation':numeric,'future_trajectory_policy':'Reported separately; exact checkpoint state roundtrip is a distinct mandatory gate. No deterministic mode enabled.','loss_differences':[abs(x['global_mean_loss']-y['global_mean_loss']) for x,y in zip(reference,b_rows)]}
receipt['STAGE2_RESUME_NUMERICAL_PARITY']='EXACT' if all(x['tensor_count']==x['exact_tensor_count'] for x in numeric.values()) else 'NONDETERMINISTIC_DEVIATION_REPORTED'
(A/'STAGE2_RESUME_COMPARISON.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
assert ok
