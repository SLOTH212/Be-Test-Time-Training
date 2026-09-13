

def th(x):return dr.tensor_hash(x)


def base(model,l):return model.model.layers[l].mlp.down_proj.weight


class PrefixKVInjectCache(TTTDynamicCache):
 def __init__(self,config=None,donor_prefix=None,prefix_len=0):
  super().__init__(config=config);self.donor_prefix=donor_prefix or {};self.prefix_len=int(prefix_len);self.replacement_records=[]
 def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
  k,v=super().update(key_states,value_states,layer_idx,cache_kwargs)
  if key_states.shape[-2]>1 and layer_idx in self.donor_prefix:
   dk,dv=self.donor_prefix[layer_idx];n=self.prefix_len
   if dk.shape[-2]!=n or dv.shape[-2]!=n:raise RuntimeError(f'DONOR_PREFIX_SHAPE:L{layer_idx}')
   k[..., :n, :].copy_(dk);v[..., :n, :].copy_(dv)
   self.replacement_records.append({'layer':int(layer_idx),'prefix_len':n,'key_exact':th(k[...,:n,:])==th(dk),'value_exact':th(v[...,:n,:])==th(dv)})
  return k,v


def install_formal(model):
 for layer in LAYERS:
  m=model.model.layers[layer].mlp;m._cl_sequence=None;m._cl_tau=None;m._cl_inject_box=None;m._cl_boundary=None;m._cl_events=[];m._cl_final=None;m._cl_trace_callback=None
  def patched(self,x,t=None,past_w=None,enabled=True):
   h=self.act_fn(self.gate_proj(x))*self.up_proj(x)
   if not hasattr(self,'ttt_conv') or not enabled:return self.down_proj(h)
   present=self.down_proj.weight if past_w is None else past_w
   if t is None:return F.linear(h,present,self.down_proj.bias),present
   full_len=(x.shape[1]//self.ttt_chunk)*self.ttt_chunk;n=full_len//self.ttt_chunk
   if self._cl_sequence is None or len(self._cl_sequence)!=n:raise RuntimeError(f'SEQUENCE_LENGTH:L{self.layer_idx}:{n}')
   state=TTTState(present);outs=[];events=[];proj=self.ttt_proj.weight
   for j,start in enumerate(range(0,full_len,self.ttt_chunk)):
    if j==self._cl_tau:
     pre=state.weight.detach().clone()
     if self._cl_inject_box is not None:state=TTTState(self._cl_inject_box['weight'])
     self._cl_boundary={'before':pre,'after':state.weight.detach().clone()}
    end=start+self.ttt_chunk;action=self._cl_sequence[j];active=int(self.layer_idx) in ACTION_LAYERS[action];pre=state.weight
    if active:
     out,proposal,stat=ttt_chunk_step(h[:,start:end],t[:,start:end],self.down_proj.weight,proj,self.ttt_conv.weight,torch.ones((1,self.ttt_chunk),dtype=torch.bool,device=x.device),state,self.ttt_lr,True,self.ttt_update_clip_norm,self.ttt_target_type,getattr(self,'ttt_ntp_gate',None));state=proposal
    else:
     out=F.linear(h[:,start:end],state.weight,self.down_proj.bias);stat=TTTStats(0,self.ttt_chunk-1,0.,0.,False)
    delta=(state.weight-pre).detach()
    events.append({'chunk_index':j+1,'action':action,'active':active,'pre_state':pre.detach(),'post_state':state.weight.detach(),'delta':delta,'raw_norm':float(stat.pre_clip_norm),'clipped_norm':float(stat.post_clip_norm)})
    if self._cl_trace_callback is not None and j==self._cl_tau:
     self._cl_trace_callback(int(self.layer_idx),'ttt_target',t[:,start:end]);self._cl_trace_callback(int(self.layer_idx),'mlp_output',out)
     if active:
      valid=torch.ones((1,self.ttt_chunk),dtype=torch.bool,device=x.device);target,vp=build_chunk_local_ntp_pairs(t[:,start:end],valid);target=target*self.ttt_ntp_gate.to(target.dtype);mask=vp.to(target.dtype).unsqueeze(-1);source=h[:,start:end-1]*vp.to(h.dtype).unsqueeze(-1);raw=torch.einsum('bti,btd,de->ei',source,target*mask,self.ttt_proj.weight)*self.ttt_lr
     else:raw=torch.zeros_like(self.down_proj.weight)
     self._cl_trace_callback(int(self.layer_idx),'raw_delta',raw);self._cl_trace_callback(int(self.layer_idx),'clipped_delta',delta);self._cl_trace_callback(int(self.layer_idx),'post_update_fast_weight',state.weight)
    outs.append(out)
   if self._cl_tau==n:
    pre=state.weight.detach().clone()
    if self._cl_inject_box is not None:state=TTTState(self._cl_inject_box['weight'])
    self._cl_boundary={'before':pre,'after':state.weight.detach().clone()}
   if full_len<x.shape[1]:outs.append(F.linear(h[:,full_len:],state.weight,self.down_proj.bias))
   self._cl_events=events;self._cl_final=state.weight.detach().clone();return torch.cat(outs,1),state.weight
  m.forward=types.MethodType(patched,m)


def configure(model,sequence,tau,inject,trace_callback=None):
 used=set(x for a in sequence for x in ACTION_LAYERS[a])|set(inject or {})
 stashed={}
 for l in LAYERS:
  m=model.model.layers[l].mlp;m._cl_sequence=list(sequence);m._cl_tau=tau;m._cl_inject_box=None if inject is None or l not in inject else {'weight':inject[l]};m._cl_boundary=None;m._cl_events=[];m._cl_final=None;m._cl_trace_callback=trace_callback
  if l not in used:
   stashed[l]={}
   for name in ('ttt_conv','ttt_proj','ttt_ntp_gate'):
    if hasattr(m,name):stashed[l][name]=getattr(m,name);delattr(m,name)
 return used,stashed


def restore(model,stashed):
 for l,attrs in stashed.items():
  for n,v in attrs.items():setattr(model.model.layers[l].mlp,n,v)


@torch.inference_mode()
def run_branch(ctx,model,tok,row,sequence,tau,inject=None,donor_prefix=None,trace_callback=None,return_cache=False):
 used,stashed=configure(model,sequence,tau,inject,trace_callback);old=model.model.ttt_mode
 if not used:model.model.ttt_mode=False
 enc=tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False);ids=enc.input_ids.cuda();n=ids.shape[1];mask=torch.ones((1,n),dtype=torch.long,device='cuda')
 cache=PrefixKVInjectCache(config=model.config,donor_prefix=donor_prefix,prefix_len=tau*1024)
 try:
  out=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
 finally:
  model.model.ttt_mode=old;restore(model,stashed)
 boundary={l:(model.model.layers[l].mlp._cl_boundary['after'] if l in used else base(model,l)) for l in LAYERS};final={l:(model.model.layers[l].mlp._cl_final if l in used else base(model,l)) for l in LAYERS}
 prompt_cache={i:(cache.layers[i].keys.detach().clone(),cache.layers[i].values.detach().clone()) for i in range(len(cache.layers))}
 prompt_ttt_occupancy={l:cache.ttt_states[l][2] is not None for l in LAYERS};prompt_ttt_hash={l:(th(cache.ttt_states[l][2]) if cache.ttt_states[l][2] is not None else None) for l in LAYERS};generation_trace=[];generated=[];next_id=int(out.logits[0,-1].argmax());generation_trace.append({'generation_step':1,'next_token_id':next_id,'logits_hash':th(out.logits),'logits_max':float(out.logits.max()),'logits_argmax':next_id})
 for _ in range(int(row['max_new_tokens'])):
  generated.append(next_id)
  if next_id==tok.eos_token_id:break
  one=torch.tensor([[next_id]],device='cuda');am=torch.ones((1,n+len(generated)),dtype=torch.long,device='cuda');out=model(input_ids=one,attention_mask=am,past_key_values=cache,use_cache=True,logits_to_keep=1);next_id=int(out.logits[0,-1].argmax());generation_trace.append({'generation_step':len(generated)+1,'next_token_id':next_id,'logits_hash':th(out.logits),'logits_max':float(out.logits.max()),'logits_argmax':next_id})
 pred=tok.decode(generated,skip_special_tokens=True);score=float(ctx.score_fn(row['family'],pred,row['reference']))
 result={'score':score,'prediction_hash':hashlib.sha256(pred.encode()).hexdigest(),'prompt_token_n':n,'input_ids_hash':hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),'attention_mask_hash':hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest(),'boundary':boundary,'final':final,'prompt_cache':prompt_cache if return_cache else None,'prompt_ttt_occupancy':prompt_ttt_occupancy,'prompt_ttt_hash':prompt_ttt_hash,'generation_trace':generation_trace,'cache_replacement_records':cache.replacement_records,'generation_update_count':0}
 return result


def donor_prefix(cache,tau):return {l:(kv[0][...,:tau*1024,:].detach().clone(),kv[1][...,:tau*1024,:].detach().clone()) for l,kv in cache.items()}
