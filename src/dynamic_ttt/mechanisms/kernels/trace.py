

def cmp(native,condition,th):
 import torch
 a=native.float(); b=condition.float(); d=a-b
 return {'exact_equal':bool(torch.equal(native,condition)),'max_abs_diff':float(d.abs().max()) if d.numel() else 0.0,'relative_l2_diff':float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(a)+1e-30)),'native_hash':th(native),'condition_hash':th(condition)}


class Recorder:
 def __init__(self,prefix,T,n,mode,th,ref=None):
  self.start=prefix*1024; self.full=T*1024; self.n=n; self.mode=mode; self.th=th; self.ref={} if ref is None else ref; self.rows=[]; self.handles=[]; self.order=0
 def record(self,key,layer,tensor,region,chunk,x):
  import torch
  if not torch.is_tensor(x): return
  x=x.detach().cpu().clone()
  if self.mode=='native': self.ref[key]=x
  elif key in self.ref:
   self.rows.append({'stage_order':self.order,'region':region,'chunk_index':chunk,'layer':layer,'tensor':tensor,**cmp(self.ref[key],x,self.th)}); self.order+=1
 def first(self,layer,tensor,x):
  import torch
  if not torch.is_tensor(x) or x.ndim<2 or x.shape[1]<=self.start:return
  self.record(f'first:{layer}:{tensor}',layer,tensor,'FIRST_CONTINUATION_CHUNK',self.start//1024+1,x[:,self.start:min(self.start+1024,x.shape[1])])
 def tail(self,layer,tensor,x):
  import torch
  if not torch.is_tensor(x) or x.ndim<2 or self.n<=self.full or x.shape[1]<=self.full:return
  self.record(f'tail:{layer}:{tensor}',layer,tensor,'INCOMPLETE_PROMPT_TAIL','TAIL',x[:,self.full:self.n])
 def internal(self,layer,tensor,x): self.record(f'internal:{layer}:{tensor}',layer,tensor,'FIRST_POST_BOUNDARY_UPDATE',self.start//1024+1,x)
 def hooks(self,m):
  for li,layer in enumerate(m.model.layers):
   self.handles.append(layer.input_layernorm.register_forward_pre_hook(lambda q,i,l=li:self.first(l,'attention_input',i[0])))
   self.handles.append(layer.self_attn.q_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'q_projection',o)))
   self.handles.append(layer.self_attn.k_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'k_projection',o)))
   self.handles.append(layer.self_attn.v_proj.register_forward_hook(lambda q,i,o,l=li:self.first(l,'v_projection',o)))
   def ah(q,i,o,l=li): self.first(l,'attention_output',o[0]); self.tail(l,'attention_output',o[0])
   self.handles.append(layer.self_attn.register_forward_hook(ah))
   self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda q,i,l=li:self.first(l,'mlp_input_residual',i[0])))
   def mh(q,i,o,l=li):
    x=o[0] if isinstance(o,tuple) else o; self.first(l,'mlp_output',x); self.tail(l,'mlp_output',x)
   self.handles.append(layer.mlp.register_forward_hook(mh))
   def lh(q,i,o,l=li): self.first(l,'residual_stream_output',o); self.tail(l,'residual_stream_output',o)
   self.handles.append(layer.register_forward_hook(lh))
  def nh(q,i,o):
   for j in range(self.start//1024,self.full//1024): self.record(f'complete:{j}:final_hidden',28,'final_complete_chunk_hidden','COMPLETE_CONTINUATION_CHUNK',j+1,o[:,j*1024:(j+1)*1024])
   if self.n>self.full:self.record('tail:28:final_hidden',28,'final_tail_hidden','INCOMPLETE_PROMPT_TAIL','TAIL',o[:,self.full:self.n])
  self.handles.append(m.model.norm.register_forward_hook(nh))
 def close(self):
  for h in self.handles:h.remove()


def prefix_hashes(cache,prefix,th):
 n=prefix*1024
 return {str(l):{'key':th(k[...,:n,:]),'value':th(v[...,:n,:])} for l,(k,v) in cache.items()}


def first_divergence(trace,l4,l5):
 canonical_regions=['FIRST_CONTINUATION_CHUNK','FIRST_POST_BOUNDARY_UPDATE','COMPLETE_CONTINUATION_CHUNK','INCOMPLETE_PROMPT_TAIL']
 for region in canonical_regions:
  q=[x for x in trace if x['region']==region and not x['exact_equal']]
  if q:
   x=q[0]; return {'closure_level':{'FIRST_CONTINUATION_CHUNK':'L1','FIRST_POST_BOUNDARY_UPDATE':'L1','COMPLETE_CONTINUATION_CHUNK':'L2','INCOMPLETE_PROMPT_TAIL':'L3'}[region],'region':region,'layer':x['layer'],'tensor':x['tensor'],'max_abs_diff':x['max_abs_diff'],'relative_l2_diff':x['relative_l2_diff']}
 if not l4:return {'closure_level':'L4','region':'GENERATION_START','layer':-1,'tensor':'generation_start_logits'}
 if not l5:return {'closure_level':'L5','region':'FULL_OUTPUT','layer':-1,'tensor':'prediction_sha256'}
 return {'closure_level':'NONE_OBSERVED','region':'NONE_OBSERVED','layer':None,'tensor':None}
