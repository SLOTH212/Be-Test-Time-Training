"""Architecture adapter around immutable Qwen K/P prompt-chain operations.

The original layer-major MLP loop, prefix KV injection, used-layer occupancy,
and prompt runner are compiled unchanged with Llama core/cache globals.
Single-update deletion is the explicitly requested V1 engineering extension.
"""
import sys,hashlib,gc,types,time
from dynamic_ttt.mechanisms.experimental import llama_contracts as m
def tensor_hash(x):
 if x is None:return None
 import torch
 return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
class FastWeights:
 def __init__(self,model):self.model=model
 def base(self,l):return self.model.model.layers[l].mlp.down_proj.weight
 def snapshot(self,cache):return {l:None if cache.ttt_states[l][2] is None else cache.ttt_states[l][2].detach().clone() for l in m.LAYERS}
 def fingerprint(self,s):return {str(l):tensor_hash(self.base(l) if s[l] is None else s[l]) for l in m.LAYERS}
 def restore(self,cache,s,layers=None):
  for l in m.LAYERS if layers is None else layers:
   if l not in m.LAYERS:raise ValueError('UNKNOWN_STATE_LAYER')
   h,t,_=cache.ttt_states[l];cache.ttt_states[l]=(h,t,None if s[l] is None else s[l].detach().clone())
 def reset(self,cache,layers=None):self.restore(cache,{l:None for l in m.LAYERS},layers)
class Backend:
 def __init__(self,cfg,rank):
  import torch
  from dynamic_ttt.models.family import build
  from dynamic_ttt.ttt import state as core
  from transformers import AutoTokenizer
  from torch.distributed.checkpoint import FileSystemReader,load
  self.torch=torch;self.rank=rank;torch.cuda.set_device(rank);torch.manual_seed(42)
  self.model,_=build(cfg['model_path'],model_family='llama',ttt_layers=m.LAYERS,chunk_size=4096,inference=True,device='cpu')
  state=self.model.state_dict();reader=FileSystemReader(cfg['debug_dcp']);md=reader.read_metadata()
  assert {k[6:] for k in md.state_dict_metadata if k.startswith('model.')}==set(state)
  for k,v in state.items():assert tuple(md.state_dict_metadata['model.'+k].size)==tuple(v.shape)
  load({'model':state},storage_reader=reader,no_dist=True);self.model.load_state_dict(state,strict=True);del state
  self.model.to('cuda:'+str(rank)).eval().requires_grad_(False)
  assert sum(p.numel() for p in self.model.parameters())==8131072000
  self.tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
  self.fw=FastWeights(self.model);self.base_hash=self.parameter_hash();self.versions={n:p._version for n,p in self.model.named_parameters()}
  cache_cls=__import__(type(self.model).__module__,fromlist=['TTTDynamicCache']).TTTDynamicCache
  self.cache_cls=cache_cls;self.calls=[];self.delete=None;self.sequence=[];self.counters={};self.core=core
  def step(*args,**kwargs):
   layer=next(l for l in m.LAYERS if args[2] is self.fw.base(l));j=self.active_indices[layer][self.counters.get(layer,0)];self.counters[layer]=self.counters.get(layer,0)+1
   current=args[6].weight;before=tensor_hash(current)
   if self.delete==(j,layer):out=torch.nn.functional.linear(args[0],current);new=args[6];stat=core.TTTStats(0,4095,0.,0.,False)
   else:out,new,stat=core.ttt_chunk_step(*args,**kwargs)
   assert torch.equal(out,torch.nn.functional.linear(args[0],current))
   self.calls.append({'layer':layer,'chunk':j,'action':self.sequence[j],'omitted':self.delete==(j,layer),'updates':stat.update_count,'state_before':before,'state_after':tensor_hash(new.weight),'output_hash':tensor_hash(out),'clip_norm':float(stat.post_clip_norm),'real_forward':True})
   return out,new,stat
  env={'torch':torch,'F':torch.nn.functional,'dr':types.SimpleNamespace(tensor_hash=tensor_hash),'TTTDynamicCache':cache_cls,'TTTState':core.TTTState,'TTTStats':core.TTTStats,'ttt_chunk_step':step,'build_chunk_local_ntp_pairs':core.build_chunk_local_ntp_pairs,'LAYERS':m.LAYERS,'ACTION_LAYERS':m.MAPPING}
  self.c=m.rt.source_functions('kernels/closure.py',['th','base','PrefixKVInjectCache','install_formal','configure','restore','donor_prefix'],env,chunk=4096)
  self.f=m.rt.source_functions('kernels/factorial.py',['run_prompt'],{'torch':torch,'c':self.c,'LAYERS':m.LAYERS,'base':self.c.base,'th':self.c.th},chunk=4096)
  self.c.install_formal(self.model)
 def parameter_hash(self):return m.rt.canonical({n:tensor_hash(p) for n,p in self.model.named_parameters()})
 def ids(self,n):
  text='A quiet laboratory records neutral observations. The table contains a blue cup, a notebook, and a small lamp. '
  pool=self.tok(text,add_special_tokens=False)['input_ids'];return ([self.tok.bos_token_id]+pool*((n//len(pool))+1))[:n]
 def run(self,seq,*,tau=1,inject=None,donor=None,delete=None,retain=False,tail=0):
  torch=self.torch;self.sequence=list(seq);self.calls=[];self.counters={};self.delete=delete
  self.active_indices={l:[j for j,a in enumerate(seq) if l in m.MAPPING[a]] for l in m.LAYERS}
  ids=self.ids(len(seq)*4096+tail);self.initial={str(l):tensor_hash(self.fw.base(l)) for l in m.LAYERS};start=time.time()
  class Tokens:
   eos_token_id=self.tok.eos_token_id
   def __call__(obj,*a,**k):return types.SimpleNamespace(input_ids=torch.tensor([ids]))
   def decode(obj,*a,**k):return self.tok.decode(*a,**k)
  # Request diagnostic logits at each chunk end without changing model forwards.
  class ModelProxy:
   def __getattr__(obj,k):return getattr(self.model,k)
   def __call__(obj,*a,**k):
    if k['input_ids'].shape[1]>1:k['logits_to_keep']=torch.tensor([4096*(j+1)-1 for j in range(len(seq))]+([len(ids)-1] if tail else []),device='cuda')
    return self.model(*a,**k)
  with torch.inference_mode():
   x=self.f.run_prompt(None,ModelProxy(),Tokens(),{'input':'tokenized_neutral_fixture','max_new_tokens':getattr(self,'max_new_tokens',2)},seq,tau,inject=inject,donor=donor,return_cache=True,generate=True)
  ev=[]
  for l in m.LAYERS:
   mlp=self.model.model.layers[l].mlp
   ev.extend({'layer':l,'chunk':e['chunk_index']-1,'action':e['action'],'active':e['active'],'state_before':tensor_hash(e['pre_state']),'state_after':tensor_hash(e['post_state'])} for e in mlp._cl_events)
  compact={'sequence':list(seq),'prompt_ids_sha256':m.rt.canonical(ids),'prompt_token_count':len(ids),'chunk_boundaries':[4096*j for j in range(len(seq)+1)],'initial':self.initial,'final':{str(l):tensor_hash(v) for l,v in x['final'].items()},'boundary':{str(l):tensor_hash(v) for l,v in x['boundary'].items()},'occupancy':{str(l):v for l,v in x['prompt_ttt_occupancy'].items()},'prefix_hashes':{str(l):[tensor_hash(k[...,:tau*4096,:]),tensor_hash(v[...,:tau*4096,:])] for l,(k,v) in x['prompt_cache'].items()},'logits_hash':x['prompt_logits_hash'],'chunk_logits_hashes':[tensor_hash(x['prompt_logits'][:,j]) for j in range(x['prompt_logits'].shape[1])],'generation_trace':x['generation_trace'],'prediction_hash':x['prediction_hash'],'generation_updates':x['generation_update_count'],'updates':sum(e['updates'] for e in self.calls),'calls':list(self.calls),'state_events':ev,'kv_replacement_records':x['cache_replacement_records'],'wall_seconds':time.time()-start,'downstream_recomputation':True,'stored_delta_stitching':False}
  assert all(p._version==self.versions[n] for n,p in self.model.named_parameters())
  assert compact['updates']==sum(len(m.MAPPING[a]) for a in seq)-(1 if delete else 0)
  # Drop historical dense diagnostic event references after compact hashes exist.
  for l in m.LAYERS:
   mlp=self.model.model.layers[l].mlp;mlp._cl_events=[];mlp._cl_final=None;mlp._cl_boundary=None;mlp._cl_inject_box=None
  if retain:
   kept={'boundary':{l:v.detach().clone() for l,v in x['boundary'].items()},'donor':self.c.donor_prefix(x['prompt_cache'],tau)}
  else:kept=None
  del x;gc.collect();torch.cuda.empty_cache();return compact,kept
 def adapter_test(self):
  torch=self.torch;cache=self.cache_cls(config=self.model.config)
  with torch.inference_mode():
   for l in m.LAYERS:cache.ttt_states[l]=(None,None,self.fw.base(l).clone())
   snap=self.fw.snapshot(cache);before=self.fw.fingerprint(snap);self.fw.reset(cache,[30]);assert cache.ttt_states[30][2] is None
   assert all(cache.ttt_states[l][2] is not None for l in m.LAYERS if l!=30)
   self.fw.restore(cache,snap);assert self.fw.fingerprint(self.fw.snapshot(cache))==before
   self.fw.reset(cache);assert all(cache.ttt_states[l][2] is None for l in m.LAYERS)
  del snap,cache;gc.collect();torch.cuda.empty_cache();return 'PASS'
