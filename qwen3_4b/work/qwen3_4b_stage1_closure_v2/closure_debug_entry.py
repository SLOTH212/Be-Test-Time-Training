import os,sys,json,hashlib,runpy
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_closure_v2'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
assert not torch.are_deterministic_algorithms_enabled()
sys.path.insert(0,str(R/'src/training_runtime/code'))
import lib.distributed_checkpoint as cp
from lib.pipeline_core import load_yaml
cfg=load_yaml(sys.argv[sys.argv.index('--config')+1]);assert not cfg.get('formal_training')
out=Path(sys.argv[sys.argv.index('--output-dir')+1]);assert out.resolve().is_relative_to(R)
out.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['RANK'])
def normal(x):
 if isinstance(x,torch.Tensor):
  local=x.to_local() if hasattr(x,'to_local') else x
  t=local.detach().cpu().contiguous();b=t.reshape(-1).view(torch.uint8).numpy()
  return {'tensor':True,'dtype':str(t.dtype),'shape':list(t.shape),'global_shape':list(x.shape),'sha256':hashlib.sha256(b.tobytes()).hexdigest(),'placement':str(getattr(x,'placements',None))}
 if isinstance(x,np.ndarray):return {'numpy':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
 if isinstance(x,dict):return {str(k):normal(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [normal(v) for v in x]
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 raise TypeError(str(type(x)))
def snapshot(model,opt,progress,scheduler):
 names={id(p):n for n,p in model.named_parameters()}
 return {'model':{n:normal(p) for n,p in model.named_parameters()},'buffers':{n:normal(p) for n,p in model.named_buffers()},'optimizer':{names[id(p)]:normal(s) for p,s in opt.state.items()},'param_groups':[{k:([names[id(p)] for p in v] if k=='params' else normal(v)) for k,v in g.items()} for g in opt.param_groups],'rank_state':normal(cp.capture_rank_state(progress,scheduler)),'runtime':{'rank':rank,'world_size':dist.get_world_size(),'visible_device_count':torch.cuda.device_count(),'physical_gpu':2+rank,'grad_scaler':None,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'training':model.training,'scaler_reason':'BF16 fused AdamW, no GradScaler in trainer'}}
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
Base=cp.DistributedCheckpointRotation
if os.environ.get('STAGE1_CLOSURE_USE_REPAIR')=='1':
 from lib.stage1_closure_checkpoint_v2 import Stage1ClosureCheckpointRotation
 Base=Stage1ClosureCheckpointRotation
class AuditRotation(Base):
 def load(self,model,optimizer,scheduler=None):
  latest=self.latest()
  if latest is None:
   v=snapshot(model,optimizer,{},scheduler);write(out/f'initial_rank{rank}.json',v)
   return None
  result=super().load(model,optimizer,scheduler)
  state=snapshot(model,optimizer,result,scheduler)
  expected_path=self.root/f'presave_step{result["update_step"]}_rank{rank}.json'
  expected=json.loads(expected_path.read_text())
  differences=[k for k in expected if state.get(k)!=expected[k]]
  report={'status':'PASS' if not differences else 'FAIL','differences':differences,'expected_optimizer_states':len(expected['optimizer']),'loaded_optimizer_states':len(state['optimizer']),'extra_optimizer_names':sorted(set(state['optimizer'])-set(expected['optimizer'])),'rank':rank,'step':result['update_step'],'actual_loaded_snapshot':state,'expected_snapshot':str(expected_path)}
  write(out/f'load_completeness_rank{rank}.json',report)
  bad=torch.tensor(int(bool(differences)),device=rank);dist.all_reduce(bad,op=dist.ReduceOp.MAX)
  if int(bad.item()):raise RuntimeError('STAGE1_EXACT_LOAD_STATE_COMPLETENESS_FAILED')
  # Audit collectives and snapshots must not perturb restored RNG.
  saved=torch.load(self.root/latest['slot']/f'rank_{rank:05d}.pt',map_location='cpu',weights_only=False);cp.restore_rank_state(saved)
  return result
 def save(self,model,optimizer,progress,scheduler=None):
  before=snapshot(model,optimizer,progress,scheduler);path=self.root/f'presave_step{progress["update_step"]}_rank{rank}.json';write(path,before)
  fail_step=int(os.environ.get('STAGE1_CLOSURE_FAIL_DURING_SAVE_STEP','0'))
  import torch.distributed.checkpoint as dcp
  original_save=dcp.save
  if fail_step==progress['update_step']:
   def injected(*args,**kwargs):
    partial=Path(kwargs['checkpoint_id']);partial.mkdir(exist_ok=True)
    torch.save({'DEBUG_ONLY':True,'rank':rank,'step':progress['update_step'],'injected_partial':True},partial/f'injected_rank{rank}.pt')
    print('CONTROLLED_SOFTWARE_FAILURE_DURING_PARTIAL_DCP rank='+str(rank),flush=True)
    dist.barrier()
    if rank==1:raise RuntimeError('CONTROLLED_DEBUG_PARTIAL_SAVE_FAILURE')
    dist.barrier()
   dcp.save=injected
  try:result=super().save(model,optimizer,progress,scheduler)
  finally:dcp.save=original_save
  after=snapshot(model,optimizer,progress,scheduler)
  same=before==after;write(out/f'save_preservation_step{progress["update_step"]}_rank{rank}.json',{'status':'PASS' if same else 'FAIL','save_preserves_raw_model_optimizer_groups_rng':same})
  assert same,'DCP_SAVE_MUTATED_RAW_TRAINER_STATE'
  return result
cp.DistributedCheckpointRotation=AuditRotation
runpy.run_path(str(R/'src/training_runtime/code/workers/distributed_train_worker.py'),run_name='__main__')
