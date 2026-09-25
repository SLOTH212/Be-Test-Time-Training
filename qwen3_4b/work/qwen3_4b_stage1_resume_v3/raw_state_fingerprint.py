import os
import numpy as np
import torch
import torch.distributed as dist
import lib.distributed_checkpoint as cp
rank=int(os.environ.get('RANK','0'))
import hashlib
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
