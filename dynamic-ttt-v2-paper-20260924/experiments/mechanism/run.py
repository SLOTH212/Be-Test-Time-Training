import os,sys
W=int(sys.argv[2]) if len(sys.argv)>2 else None
os.sched_setaffinity(0,set(range(8+8*W,16+8*W)) if W is not None else set(range(8,24)))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
from pathlib import Path
import json,hashlib,types,subprocess,traceback,fcntl,time,gc
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'runtime'))
import executor as ex
import scorer
P=json.loads((R/'PROTOCOL.json').read_text());C=json.loads((R/'COHORT.json').read_text());PH=ex.sha(R/'PROTOCOL.json')
def put(p,x):ex.atomic_json(p,x)
def key(s):return hashlib.sha256(s.encode()).hexdigest()
def gate():
 for n,h in json.loads((R/'SOURCE_HASHES.json').read_text()).items():assert ex.sha(R/n)==h,n
 for c in C:assert ex.sha(Path(c['dynamic_source']['path']))==c['dynamic_source']['sha256']
def load():
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(P['seed']);torch.cuda.manual_seed_all(P['seed'])
 ctx=types.SimpleNamespace(code=R/'runtime/code',ckpt=Path(P['model_path']),run=R,score_fn=scorer.per_sample_score,training_chunk=1024,chunk=1024,mechanism_cache=None)
 m,tok=ex.load_model(ctx);rows={x['sample_id']:x for x in ex.load_jsonl(R/'samples.jsonl')};return ctx,m,tok,rows

def eval_branch(ctx,m,tok,row,seq,tau=None,reset=(),inject=None,cache=None,trace=False,label='branch'):
 import torch
 ctx.mechanism_cache=cache;handles=[];tr={};boundary=tau*1024 if tau is not None else 0
 for l in ex.LAYERS:m.model.layers[l].mlp._mechanism=dict(tau=tau,reset_layers=list(reset),inject=inject or {})
 def rec(name,x,full=False):
  if not torch.is_tensor(x) or x.ndim<3 or x.shape[1]<=1:return
  y=x[:,boundary:] if full else x[:,boundary:boundary+1024]
  tr[name]=ex.tensor_hash(y)
  if not full and x.shape[1]%1024:tr[name+':tail']=ex.tensor_hash(x[:,-(x.shape[1]%1024):])
 if trace:
  for i,layer in enumerate(m.model.layers):
   handles.append(layer.input_layernorm.register_forward_pre_hook(lambda q,args,l=i:rec(f'{l}:attention_input',args[0])))
   for name in ['q_proj','k_proj','v_proj']:handles.append(getattr(layer.self_attn,name).register_forward_hook(lambda q,args,out,l=i,n=name:rec(f'{l}:{n}',out)))
   handles.append(layer.self_attn.register_forward_hook(lambda q,args,out,l=i:rec(f'{l}:attention_output',out[0])))
   handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda q,args,l=i:rec(f'{l}:mlp_input',args[0])))
   handles.append(layer.mlp.register_forward_hook(lambda q,args,out,l=i:rec(f'{l}:mlp_output',out[0] if isinstance(out,tuple) else out)))
   handles.append(layer.register_forward_hook(lambda q,args,out,l=i:rec(f'{l}:residual',out[0] if isinstance(out,tuple) else out)))
  handles.append(m.model.norm.register_forward_hook(lambda q,args,out:rec('final_suffix_hidden',out,True)))
  def logits(q,args,out):
   if 'prompt_logits' not in tr:tr['prompt_logits']=ex.tensor_hash(out.logits)
  handles.append(m.register_forward_hook(logits))
 try:
  ev=ex.Evaluator(ctx,m,tok,row,label,1,1,R/f'candidate_{W}.json');out=ev.evaluate(seq,capture=False,force=True);ev.finish()
  events=[e for l in ex.LAYERS for e in m.model.layers[l].mlp._intervention_events];weights={l:m.model.layers[l].mlp._boundary_before for l in ex.LAYERS} if tau is not None else {}
  out['interventions']=events;out['trace']=tr;out['base_unchanged']=True
  if tau is not None:
   assert len(events)==5 and all(e['chunk_index']==tau+1 for e in events)
   for e in events:
    if e['reset']:assert e['after']==ex.tensor_hash(m.model.layers[e['layer']].mlp.down_proj.weight)
    if e['injected']:assert e['after']==ex.tensor_hash(inject[e['layer']])
  return out,weights
 finally:
  for h in handles:h.remove()
  ctx.mechanism_cache=None
  for l in ex.LAYERS:m.model.layers[l].mlp._mechanism={}

def cache_class():
 from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
 class Cache(TTTDynamicCache):
  def __init__(self,config,prefix,donor=None):super().__init__(config=config);self.prefix=prefix;self.donor=donor or {};self.replacements=[]
  def update(self,k,v,layer_idx,cache_kwargs=None):
   k,v=super().update(k,v,layer_idx,cache_kwargs)
   if k.shape[-2]>1 and layer_idx in self.donor and not any(x['layer']==layer_idx for x in self.replacements):
    dk,dv=self.donor[layer_idx];n=self.prefix;assert dk.shape[-2]==dv.shape[-2]==n
    k[...,:n,:].copy_(dk.to(k.device));v[...,:n,:].copy_(dv.to(v.device));assert ex.tensor_hash(k[...,:n,:])==ex.tensor_hash(dk) and ex.tensor_hash(v[...,:n,:])==ex.tensor_hash(dv)
    self.replacements.append(dict(layer=layer_idx,prefix_tokens=n,key_exact=True,value_exact=True))
   return k,v
 return Cache

def parity(z,c):assert z['score']==c['native_score'] and z['prediction_hash']==c['native_prediction_hash'],('NATIVE_PARITY',c['sample_id'])
def core_one(ctx,m,tok,row,c):
 seq=c['sequence'];tau=c['tau'];native,_=eval_branch(ctx,m,tok,row,seq,label='native');parity(native,c)
 reverse,_=eval_branch(ctx,m,tok,row,list(reversed(seq)),label='reverse')
 result=dict(sample_id=c['sample_id'],task=c['task'],protocol_sha256=PH,native=native,reverse=reverse,reset=None,deletion=[],sham=None,K=None)
 if c['reset_eligible']:
  reset,_=eval_branch(ctx,m,tok,row,seq,tau,ex.LAYERS,label='reset');result['reset']=reset
  for l in c['deletion_targets']:
   z,_=eval_branch(ctx,m,tok,row,seq,tau,[l],label=f'deletion_L{l}');result['deletion'].append(dict(layer=l,result=z))
  if c['sham_layer'] is not None:
   z,_=eval_branch(ctx,m,tok,row,seq,tau,[c['sham_layer']],label='sham');parity(z,c);result['sham']=z
  Cache=cache_class();nc=Cache(m.config,tau*1024);nt,weights=eval_branch(ctx,m,tok,row,seq,tau,cache=nc,trace=True,label='K_native');parity(nt,c)
  donor={i:(l.keys[...,:tau*1024,:].detach().cpu().clone(),l.values[...,:tau*1024,:].detach().cpu().clone()) for i,l in enumerate(nc.layers)}
  assert all(nc.ttt_states[l][2] is not None for l in ex.LAYERS);del nc
  controlseq=['OFF']*tau+seq[tau:];control,_=eval_branch(ctx,m,tok,row,controlseq,tau,label='control')
  k0,_=eval_branch(ctx,m,tok,row,controlseq,tau,inject=weights,trace=True,label='K0')
  kc=Cache(m.config,tau*1024,donor);k1,_=eval_branch(ctx,m,tok,row,controlseq,tau,inject=weights,cache=kc,trace=True,label='K1');parity(k1,c)
  assert len(kc.replacements)==len(m.model.layers) and all(kc.ttt_states[l][2] is not None for l in ex.LAYERS)
  # Prefix-dependent projection tensors may legitimately differ in injected-prefix regions;
  # every recorded suffix tensor and generation-start output must close exactly.
  assert k1['trace']==nt['trace'],('K1_TRACE_CLOSURE',c['sample_id'],[k for k in nt['trace'] if nt['trace'][k]!=k1['trace'].get(k)])
  result['K']=dict(native=nt,control=control,K0=k0,K1=k1,K0_trace_closure=k0['trace']==nt['trace'],K1_trace_closure=True,K0_prediction_closure=k0['prediction_hash']==nt['prediction_hash'],replacement_records=kc.replacements,all_five_materialized=True,P_omitted=True)
  del kc,donor,weights
 again,_=eval_branch(ctx,m,tok,row,seq,label='native_repeat');parity(again,c);result['native_repeat_pass']=True;result['at']=ex.now();return result

def r2_one(ctx,m,tok,row,c):
 native,_=eval_branch(ctx,m,tok,row,c['sequence'],label='r2_native');parity(native,c)
 src=json.loads(Path(c['dynamic_source']['path']).read_text());lookup={tuple(x['sequence']):x for x in src['candidates']};ev=ex.Evaluator(ctx,m,tok,row,'exact_r2',1,len(C),R/f'r2_progress_{W}.json');cands=[];reused=0;bridge=False
 for tau in range(1,c['T']):
  for ai,a in enumerate(ex.ACTIONS):
   for bi,b in enumerate(ex.ACTIONS):
    if a==b:continue
    seq=tuple([a]*tau+[b]*(c['T']-tau));old=lookup.get(seq)
    if old is not None:
     z=old;reused+=1
     if not bridge:
      replay=ev.evaluate(seq,force=True);assert replay['score']==z['score'] and replay['prediction_hash']==z['prediction_hash'];bridge=True
    else:z=ev.evaluate(seq)
    cands.append(dict(tau=tau,A=a,B=b,score=z['score'],prediction_hash=z['prediction_hash'],sequence_hash=z['sequence_hash'],reused=old is not None))
 assert len(cands)==c['r2_candidates'] and len({z['sequence_hash'] for z in cands})==len(cands)
 best=min(cands,key=lambda x:(-x['score'],x['tau'],ex.ACTIONS.index(x['A']),ex.ACTIONS.index(x['B']),x['sequence_hash']))
 seq=[best['A']]*best['tau']+[best['B']]*(c['T']-best['tau']);replay=ev.evaluate(seq,capture=False,force=True);assert replay['score']==best['score'] and replay['prediction_hash']==best['prediction_hash'];ev.finish()
 return dict(sample_id=c['sample_id'],task=c['task'],protocol_sha256=PH,native_score=c['native_score'],sample_best=c['sample_best'],off=c['off'],native_switches=c['switch_count'],r2_score=best['score'],matches_native=best['score']>=c['native_score']-1e-12,best=best,winner=replay,candidate_count=len(cands),reused=reused,executed_new=len(cands)-reused,candidates=cands,at=ex.now())

def worker(phase):
 gate();ctx,m,tok,rows=load();co=C[W::2]
 if phase=='smoke':
  # Same structurally selected samples on both CPU affinities; include ALL prefix and sham.
  pick=[]
  for pred in [lambda c:c['reset_eligible'] and c['all_prefix'],lambda c:c['reset_eligible'] and c['sham_layer'] is not None,lambda c:not c['reset_eligible']]:
   z=next((c for c in C if pred(c) and c not in pick),None)
   if z:pick.append(z)
  for c in pick:
   z=core_one(ctx,m,tok,rows[c['sample_id']],c);put(R/'smoke'/f'{W}_{key(c["sample_id"])}.json',z)
  put(R/f'SMOKE_{W}.json',dict(status='PASS',samples=len(pick),at=ex.now()));return
 assert json.loads((R/f'SMOKE_{W}.json').read_text())['status']=='PASS'
 for i,c in enumerate(co,1):
  dest=R/phase/f'{key(c["sample_id"])}.json'
  if dest.exists():assert json.loads(dest.read_text())['protocol_sha256']==PH;continue
  put(R/f'STATUS_{W}.json',dict(phase=phase,state='RUNNING',completed=i-1,total=len(co),sample_id=c['sample_id'],at=ex.now()))
  z=core_one(ctx,m,tok,rows[c['sample_id']],c) if phase=='core' else r2_one(ctx,m,tok,rows[c['sample_id']],c)
  put(dest,z);print(json.dumps(dict(worker=W,phase=phase,done=i,total=len(co),sample_id=c['sample_id'],at=ex.now())),flush=True);gc.collect()
 put(R/f'COMPLETE_{phase}_{W}.json',dict(status='COMPLETE',samples=len(co),at=ex.now()))

if __name__=='__main__':
 try:
  if W is not None:worker(sys.argv[1])
  else:
   lock=(R/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate()
   for phase in ['smoke','core','r2']:
    put(R/'STATUS.json',dict(state='RUNNING',phase=phase,at=ex.now()));children=[subprocess.Popen([sys.executable,'-u',str(R/'run.py'),phase,str(w)]) for w in range(2)]
    running={p.pid:p for p in children}
    while running:
     pid,raw=os.wait();code=os.waitstatus_to_exitcode(raw);running.pop(pid)
     if code:
      for p in running.values():p.terminate()
      for p in running.values():p.wait()
      raise RuntimeError(f'{phase} worker {pid} failed {code}')
    if phase in ['core','r2']:subprocess.run([sys.executable,str(R/'summarize.py')],check=True)
   put(R/'COMPLETE.json',dict(state='COMPLETE',at=ex.now()));put(R/'STATUS.json',dict(state='COMPLETE',at=ex.now()))
 except BaseException:
  put(R/('FAILURE.json' if W is None else f'FAILURE_{W}.json'),dict(error=traceback.format_exc(),at=ex.now()));raise
