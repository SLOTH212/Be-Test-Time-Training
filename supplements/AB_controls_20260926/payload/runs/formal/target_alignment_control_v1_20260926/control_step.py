"""Default-disabled, branch-local target-pairing intervention."""
import torch
import torch.nn.functional as F
from numerical import original_apply, calibrate, deterministic_permutation
from hf_models.hf_qwen3.ttt_state_core import TTTState, TTTStats, ttt_chunk_step, build_chunk_local_ntp_pairs, GATED_NTP_TARGET
from executor import tensor_hash

class Controller:
 def __init__(self):self.mode='disabled';self.writes=[]
 def begin(self,mode,sample_id,task,seed=None,native=None,audit_hashes=False,compare_original=False):
  self.mode=mode;self.sample_id=sample_id;self.task=task;self.seed=seed;self.native=native or {};self.audit_hashes=audit_hashes;self.compare_original=compare_original;self.writes=[]
 def step(self,hidden,target_source,base_weight,target_projection,conv_kernel,valid_mask,state=None,learning_rate=3.,enabled=True,delta_clip_norm=None,target_type='conv',target_gate=None,*,_layer,_chunk,_action):
  args=(hidden,target_source,base_weight,target_projection,conv_kernel,valid_mask,state,learning_rate,enabled,delta_clip_norm,target_type,target_gate)
  if self.mode=='disabled':return ttt_chunk_step(*args)
  assert enabled and target_type==GATED_NTP_TARGET and hidden.shape[0]==1 and hidden.shape[1]==1024
  assert hidden.dtype==target_source.dtype==base_weight.dtype==torch.bfloat16
  current=base_weight if state is None else state.weight
  output=F.linear(hidden,current) # read before constructing this chunk's update
  valid=valid_mask.bool();target,update_valid=build_chunk_local_ntp_pairs(target_source,valid)
  target=target*target_gate.to(device=target.device,dtype=target.dtype)
  update_hidden=hidden[:,:-1]
  indices=torch.nonzero(update_valid[0],as_tuple=False).flatten();n=int(indices.numel())
  assert n==1023,('UNEXPECTED_EFFECTIVE_PAIR_COUNT',n)
  if self.mode in ('B1','B2','identity'):
   if self.mode=='identity':
    perm=torch.arange(n,dtype=torch.long);pd=dict(permutation_hash='identity',n_targets=n,fixed_points=n,derived_seed=None,attempts=0,cyclic_fallback=False)
   else:
    perm,pd=deterministic_permutation(n,self.sample_id,_layer,_chunk,self.seed)
    assert pd['fixed_points']==0
   # All complete prompt chunks in this frozen evaluator have an all-valid mask.
   # Only target pairing changes; update_hidden is never permuted.
   assert bool(update_valid.all())
   target=target.index_select(1,perm.to(target.device))
  else:pd=dict(permutation_hash=None,n_targets=n,fixed_points=n)
  mask=update_valid.to(hidden.dtype).unsqueeze(-1)
  raw=torch.einsum('bti,btd,de->ei',update_hidden*mask,target*mask,target_projection)*learning_rate
  key=(_layer,_chunk);ref=None if self.mode=='native' else self.native.get(key)
  if self.mode=='B2':
   assert ref is not None
   after,clipped,cal=calibrate(raw,current,ref['native_represented_norm'],delta_clip_norm)
   pre=float(torch.linalg.vector_norm(raw.float()));post=cal['clipped_norm'];rep=cal['represented_norm'];was_clipped=cal['was_clipped']
  else:
   after,clipped,pre,post,rep,was_clipped=original_apply(raw,current,delta_clip_norm)
   targetnorm=ref['native_represented_norm'] if ref else rep
   rel=abs(rep-targetnorm)/targetnorm if targetnorm else (0. if rep==0 else None)
   cal=dict(alpha=1.,relative_error=rel,matched_within_1pct=rel is not None and rel<=.01,n_evaluations=1,iterations=0,zero_native=targetnorm==0,observedmax=rep)
  stat=TTTStats(1,int(update_valid.sum()),cal['scaled_raw_norm'] if self.mode=='B2' else pre,post,was_clipped)
  hashes={k:(tensor_hash(t) if self.audit_hashes else None) for k,t in [('raw_hash',raw),('clipped_hash',clipped),('before_hash',current),('after_hash',after),('output_hash',output)]}
  if self.compare_original:
   oo,ss,st=ttt_chunk_step(*args)
   assert torch.equal(oo,output) and torch.equal(ss.weight,after) and st==stat,('ORIGINAL_STEP_NOT_EXACT',self.sample_id,key)
  if self.mode=='identity':
   assert ref is not None
   assert all(hashes[k]==ref[k] for k in hashes),('IDENTITY_UPDATE_NOT_EXACT',self.sample_id,key)
  row=dict(sample_id=self.sample_id,task=self.task,chunk_index=_chunk,layer=_layer,action=_action,seed=self.seed,mode=self.mode,**pd,**hashes,
   native_raw_norm=ref['native_raw_norm'] if ref else pre,native_clipped_norm=ref['native_clipped_norm'] if ref else post,native_represented_norm=ref['native_represented_norm'] if ref else rep,
   shuffle_raw_norm=pre,clipped_norm=post,represented_norm=rep,scale_alpha=cal['alpha'],relative_norm_error=cal['relative_error'],matched_within_1pct=cal['matched_within_1pct'],calibration=cal,
   target_scope='this_sample_this_layer_this_complete_chunk',branch_local_targets=True,read_before_update=True,reward_used=False)
  self.writes.append(row)
  return output,TTTState(after),stat
