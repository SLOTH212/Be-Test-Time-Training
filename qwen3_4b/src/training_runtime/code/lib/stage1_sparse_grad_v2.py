"""Align FSDP TTT gradient participation with the existing global-window rule."""
import json,re
import torch
_ACTIVE=False
_INSTALLED=False
_PATTERN=re.compile(r'^model\.layers\.(0|6|12|18|24|30)\.mlp\.(ttt_proj\.weight|ttt_ntp_gate)$')
def materialize_missing(params,active):
 changed=[]
 if not active:return changed
 for p in params:
  name=getattr(p,'_param_fqn','')
  if not _PATTERN.fullmatch(name) or not hasattr(p,'_unsharded_param'):continue
  if p.unsharded_accumulated_grad is None and p.unsharded_param.grad is None:
   p.unsharded_param.grad=torch.zeros_like(p.unsharded_param)
   changed.append(name)
 return changed

def install(worker_globals):
 global _INSTALLED
 if _INSTALLED:raise RuntimeError('STAGE1_SPARSE_GRAD_ALREADY_INSTALLED')
 _INSTALLED=True
 original_count=worker_globals['count_window']
 def count_window(window,plan,rank,stage,ranges,device):
  global _ACTIVE
  if stage!=1:raise RuntimeError('STAGE1_SPARSE_GRAD_STAGE_REQUIRED')
  _ACTIVE=any(any(int(b)-int(a)>4096 for a,b in (json.loads(row['document_boundaries_qwen_json']) if isinstance(row['document_boundaries_qwen_json'],str) else row['document_boundaries_qwen_json'])) for _,row in window)
  return original_count(window,plan,rank,stage,ranges,device)
 worker_globals['count_window']=count_window
 from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
 from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
 original_post=FSDPParamGroup.post_backward
 def post_backward(self,*args):
  # Original post_backward is idempotent. Never synthesize gradients twice in
  # the final callback after a group's reduce-scatter already completed.
  if self._training_state!=TrainingState.POST_BACKWARD:
   changed=materialize_missing(self.fsdp_params,_ACTIVE)
   if changed:print('STAGE1_V2_ZERO_LOCAL_TTT_GRAD '+json.dumps({'names':changed,'global_window_active':True}),flush=True)
  return original_post(self,*args)
 FSDPParamGroup.post_backward=post_backward
