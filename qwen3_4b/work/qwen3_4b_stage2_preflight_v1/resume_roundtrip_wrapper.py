import os,sys,json,runpy,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
R=Path('/path/to/ttt');A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
sys.path.insert(0,str(R/'src/training_runtime/code'))
import lib.distributed_checkpoint as dc
from lib.stage2_checkpoint_runtime import Stage2CheckpointRotation
Original=Stage2CheckpointRotation
def exact(a,b):
 if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
class AuditedRotation(Original):
 def load(self,model,optimizer,scheduler=None):
  progress=super().load(model,optimizer,scheduler)
  if progress is None:return progress
  rank=dist.get_rank();latest=json.loads((self.root/'latest.json').read_text());slot=self.root/latest['slot']
  saved=torch.load(slot/f'rank_{rank:05d}.pt',map_location='cpu',weights_only=False)
  now=dc.capture_rank_state(progress,scheduler)
  rng_exact=exact(saved,now)
  from torch.distributed.checkpoint.state_dict import get_state_dict,StateDictOptions
  full_model,full_opt=get_state_dict(model,optimizer,options=StateDictOptions(full_state_dict=True,cpu_offload=True,strict=False))
  checks={'rank':rank,'rank_payload_scheduler_cursor_rng_exact':rng_exact,'lazy_restore':self.stage2_lazy_restore_audit}
  if rank==0:
   from torch.distributed.checkpoint import FileSystemReader
   from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
   from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
   reference={};_load_state_dict(reference,storage_reader=FileSystemReader(slot/'dcp'),planner=_EmptyStateDictLoadPlanner(),no_dist=True)
   checks['model_exact']=exact(reference['model'],full_model);checks['optimizer_exact']=exact(reference['optimizer'],full_opt)
   checks['model_keys']=len(full_model);checks['optimizer_state_keys']=len(full_opt['state'])
   del reference
  del full_model,full_opt
  rows=[None]*dist.get_world_size();dist.all_gather_object(rows,checks)
  ok=all(r['rank_payload_scheduler_cursor_rng_exact'] for r in rows) and rows[0]['model_exact'] and rows[0]['optimizer_exact']
  if rank==0:(A/'STAGE2_DCP_EXACT_STATE_ROUNDTRIP.json').write_text(json.dumps({'status':'PASS' if ok else 'FAIL','ranks':rows,'source':str(slot),'step':progress['update_step'],'no_future_trajectory_bitwise_requirement':True},indent=2)+'\n')
  if not ok:raise RuntimeError('STAGE2_DCP_STATE_ROUNDTRIP_FAILED')
  # Restore exact checkpoint RNG after audit operations, before resumed step.
  dc.restore_rank_state(saved)
  return progress
dc.DistributedCheckpointRotation=AuditedRotation
runpy.run_path(str(R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'),run_name='__main__')
