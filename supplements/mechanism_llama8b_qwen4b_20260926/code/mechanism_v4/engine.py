"""Isolated V2 improved-only mechanisms: reverse, reset, KV (no P), exact R2."""
from pathlib import Path
import os,sys,json,hashlib,time,types,ast,collections,subprocess,fcntl,traceback,argparse,math
R=Path('/path/to/ttt')
S=R/'work/mechanism_v2_7gpu_20260926'
O=R/'runs/mechanism_v2_7gpu_20260926'
GPUS=[0,1,2,3,4,6,7]
LAYERS=[0,6,12,18,24,30]
ACTIONS=['OFF','L0','L6','L12','L18','L24','L30','ALL']
MAPPING={'OFF':[],'ALL':LAYERS,**{f'L{l}':[l] for l in LAYERS}}
BENCH=R/'shared/benchmarks/ruler_32k_standard_13task500_v1'
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def mod(name,p):
 import importlib.util
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def screen(seq):
 return mod('screen',R/'src/reset_resolution_v2/candidate.py').screen(seq,ACTIONS)
def boundary(seq):
 j=len(seq)-1
 while j>0 and seq[j-1]==seq[-1]:j-=1
 return j
def candidates(n):
 return [dict(A=a,B=b,tau=t,sequence=[a]*t+[b]*(n-t),full_schedule_hash=canonical([a]*t+[b]*(n-t))) for a in ACTIONS for b in ACTIONS if a!=b for t in range(1,n)]
def prepare():
 assert not (S/'JOB.json').exists(),'IMMUTABLE_JOB_ALREADY_EXISTS'
 lc=read(R/'configs/llama31_8b_ruler32k_fixed8_4k_executor_v2.json')
 qj=read(R/'runs/qwen3_4b_ruler32k_dynamic_4k_executor_v2_gpu05/JOB.json')
 specs={
  'llama8b':dict(run=str(R/'runs/llama31_dynamic_v2_no_cwe_v1'),model=lc['model_path'],parent=lc['parent_authority'],source=lc['llama_source'],model_kind='llama',fixed=str(R/'runs/llama31_8b_ruler32k_fixed8_4k_executor_v2/AGGREGATE.json')),
  'qwen4b':dict(run=str(R/'runs/qwen3_4b_ruler32k_dynamic_4k_executor_v2_gpu05'),model=str(Path(qj['model_parent_authority']).parent),parent=qj['model_parent_authority'],source=str(R/'src/inference_runtime_4k_stage2_v1/code'),model_kind='qwen',fixed=qj['fixed_aggregate'])}
 bindings={}
 def bind(p):bindings[str(p)]=sha(p)
 for p in [S/'run.py',S/'trace_support.py',R/'src/reference_executor_v2_port/v2_functions.py',R/'src/reference_executor_v2_port/adapter.py',R/'src/reset_resolution_v2/candidate.py',R/'vendor/frozen/ttt_inference_runtime_v1/scorers/common.py',BENCH/'authority/RULER_32K_AUTHORITY.json',BENCH/'authority/RULER_32K_GENERATION_CONFIG.json',BENCH/'manifests/SAMPLE_INDEX.jsonl']:bind(p)
 index=[json.loads(l) for l in (BENCH/'manifests/SAMPLE_INDEX.jsonl').open()]
 assert len(index)==6500
 for p in {BENCH/x['official_jsonl_path'] for x in index}:bind(p)
 for label,c in specs.items():
  root=Path(c['run']);assert read(root/'PHASE_STATUS.json')['status']=='COMPLETE'
  bind(root/'DYNAMIC_FINAL_AUTHORITY.json');bind(c['parent']);bind(c['fixed'])
  pa=read(c['parent']);assert pa.get('DEBUG_ONLY') is False
  for n,h in pa['files'].items():
   p=Path(c['model'])/n;assert sha(p)==h,('MODEL_HASH',p);bindings[str(p)]=h
  for p in Path(c['source']).rglob('*.py'):bind(p)
  fixed={x['sample_id']:x for x in read(c['fixed'])['per_sample']}
  cohort=[]
  for p in sorted((root/'samples').glob('*.json')):
   d=read(p);assert d['status']=='COMMITTED' and d['payload_sha256']==canonical({k:v for k,v in d.items() if k!='payload_sha256'}),p
   x=d.get('raw',d);sid=x['sample_id'];assert abs(max(fixed[sid][a] for a in ACTIONS)-x['fixed_oracle_score'])<1e-12
   if x['dynamic_minus_fixed']<=1e-12:continue
   bind(p);seq=x['best_sequence'];assert 'cwe' not in x['task']
   cohort.append(dict(sample_id=sid,task=x['task'],sequence=seq,score=x['dynamic_search_score'],prediction_hash=x['best_prediction_hash'],sample_best=x['fixed_oracle_score'],reset_screen=screen(seq),tau=boundary(seq),source=str(p),source_sha256=sha(p),fixed_scores=x['fixed_scores'],constant_hashes={a:next(z['prediction_hash'] for z in x['candidates'] if z['sequence']==[a]*len(seq)) for a in ACTIONS}))
  assert len(cohort)==(39 if label=='llama8b' else 85)
  parts=[[] for _ in GPUS];cost=[0]*len(GPUS)
  for x in sorted(cohort,key=lambda x:len(x['sequence']),reverse=True):
   i=min(range(len(GPUS)),key=lambda i:cost[i]);parts[i].append(x['sample_id']);cost[i]+=len(candidates(len(x['sequence'])))
  c.update(cohort=cohort,partitions=parts)
  atomic(O/label/'COHORT.json',cohort)
 for n in [2,6,7]:
  cs=candidates(n);assert len(cs)==56*(n-1)==len({tuple(x['sequence']) for x in cs})
  assert all(sum(a!=b for a,b in zip(x['sequence'],x['sequence'][1:]))==1 for x in cs)
 atomic(S/'JOB.json',dict(schema='MECHANISM_V2_NO_P_V1',models=specs,bindings=bindings,gpus=GPUS,experiments=['native','reverse','reset','kv','r2'],reset_rule='resolution-aware V2: first switch <= max(2,N//4), late modal >= 0.8',kv='native boundary weights on all six layers; OFF-prefix control; compare control vs native prefix KV; no P factor',r2='all ordered distinct actions, all interior boundaries; tie tau then canonical action order',authorized=True))
 atomic(O/'CPU_PREFLIGHT.json',dict(status='PASS',models={k:dict(samples=len(v['cohort']),eligible=sum(x['reset_screen']['eligible'] for x in v['cohort'])) for k,v in specs.items()},r2_counts={n:len(candidates(n)) for n in [2,6,7]},job_sha256=sha(S/'JOB.json')))
 print('PREPARED',flush=True)
def verify(full=False):
 j=read(S/'JOB.json')
 for p,h in j['bindings'].items():
  if full or not str(p).endswith('.safetensors'):assert sha(p)==h,('INPUT_CHANGED',p)
 return j
def getrow(sid):
 idx=[json.loads(l) for l in (BENCH/'manifests/SAMPLE_INDEX.jsonl').open()];x=next(x for x in idx if x['project_sample_id']==sid)
 with (BENCH/x['official_jsonl_path']).open() as f:
  for i,line in enumerate(f,1):
   if i==x['source_line']:d=json.loads(line);break
 assert hashlib.sha256(d['prompt'].encode()).hexdigest()==x['input_sha256'] and canonical(d['answer'])==x['reference_sha256']
 tasks={x['name']:x['family'] for x in read(BENCH/'authority/RULER_32K_GENERATION_CONFIG.json')['tasks']}
 return dict(sample_id=sid,input=d['prompt'],reference=d['answer'],family=tasks[x['task']],max_new_tokens=x['tokens_to_generate'])
def tensor_hash(x):
 import torch
 return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def param_hash(model):return canonical({n:tensor_hash(p) for n,p in model.named_parameters()})
class Engine:
 def __init__(self,c):
  import torch
  from transformers import AutoTokenizer
  self.torch=torch;torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(42)
  sys.path.insert(0,c['source'])
  if c['model_kind']=='llama':
   from inference_model.hf_llama.modeling_llama import LlamaForCausalLM as Model,TTTDynamicCache
   from ntp_core import ttt_state_core as core
   config=None
  else:
   from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM as Model,TTTDynamicCache
   from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
   from hf_models.hf_qwen3 import ttt_state_core as core
   config=Qwen3Config.from_pretrained(c['model'],local_files_only=True)
   for k,v in dict(ttt_mode=True,ttt_layers=LAYERS,ttt_chunk=4096,ttt_lr=1.,ttt_target='hidden_states',ttt_target_type='gated_next_position_hidden',ttt_proj=True,ttt_ntp_gate=True,ttt_update_clip_norm=1e-5,use_cache=True).items():setattr(config,k,v)
  kwargs={} if config is None else {'config':config}
  self.model,info=Model.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True,**kwargs)
  assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
  self.model.cuda().eval().requires_grad_(False);self.core=core;self.cache_cls=TTTDynamicCache
  self.write_calls=[];self.forward_tokens=0;original_step=core.ttt_chunk_step
  def observed_step(*args,**kwargs):
   assert self.forward_tokens>1,'GENERATION_WRITE'
   out,new,stats=original_step(*args,**kwargs)
   self.write_calls.append(dict(forward_tokens=self.forward_tokens,valid_tokens=int(stats.valid_tokens),update_count=int(stats.update_count)))
   return out,new,stats
  core.ttt_chunk_step=observed_step
  self.tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);self.tok.padding_side='left'
  self.base_hash=param_hash(self.model);self.versions={n:p._version for n,p in self.model.named_parameters()}
  self.scorer=mod('scorer',R/'vendor/frozen/ttt_inference_runtime_v1/scorers/common.py').per_sample_score
  self.install()
 def install(self,original=False):
  p=R/'src/reference_executor_v2_port/v2_functions.py';source=p.read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_dynamic_forward');source=ast.unparse(node)
  source=source.replace('from hf_models.hf_qwen3.ttt_state_core import','from '+self.core.__name__+' import')
  if not original:
   needle='end = start + self.ttt_chunk'
   assert source.count(needle)==1
   # Exact V2 numerical loop; add only boundary capture/state intervention before selected chunk.
   lines=source.splitlines();i=next(i for i,l in enumerate(lines) if needle in l);indent=lines[i][:len(lines[i])-len(lines[i].lstrip())]
   extra=['if j == getattr(self, "_mech_tau", None):','    self._mech_boundary = state.weight.detach().clone()','    if self._mech_inject is not None:','        state = TTTState(self._mech_inject)','    self._mech_after = state.weight.detach().clone()']
   lines[i:i]=[indent+l for l in extra];source='\n'.join(lines)
  ns=dict(LAYERS=LAYERS,ACTION_LAYERS=MAPPING,types=types,tensor_hash=tensor_hash)
  exec(compile(source,str(p), 'exec'),ns);ns['install_dynamic_forward'](self.model)
 def run(self,row,seq,tau=None,inject=None,donor=None,retain=False,capture=False):
  t=self.torch;enc=self.tok(row['input'],return_tensors='pt',add_special_tokens=True,truncation=False);n=enc.input_ids.shape[1]
  assert n//4096==len(seq) and n+row['max_new_tokens']<=32768
  for l in LAYERS:
   m=self.model.model.layers[l].mlp;m._dynamic_sequence=list(seq);m._branch_id='mechanism';m._capture_identity=capture;m._events=[];m._generation_forward_calls=0;m._mech_tau=tau;m._mech_inject=None if inject is None else inject.get(l);m._mech_boundary=None;m._mech_after=None;m._captured_final_state=None
  cachecls=self.cache_cls;plen=(tau or 0)*4096;self.write_calls=[]
  class Cache(cachecls):
   def __init__(obj):super().__init__(config=self.model.config);obj.replacements=[]
   def update(obj,k,v,layer_idx,cache_kwargs=None):
    k,v=super().update(k,v,layer_idx,cache_kwargs)
    if donor is not None and k.shape[-2]==n:
     assert layer_idx not in [z['layer'] for z in obj.replacements],'REPEATED_KV_REPLACEMENT'
     dk,dv=donor[layer_idx];assert dk.shape[-2]==plen and dv.shape[-2]==plen
     k[...,:plen,:].copy_(dk);v[...,:plen,:].copy_(dv)
     kh=tensor_hash(k[...,:plen,:]);vh=tensor_hash(v[...,:plen,:]);assert kh==tensor_hash(dk) and vh==tensor_hash(dv)
     obj.replacements.append(dict(layer=layer_idx,prefix_tokens=plen,key_hash=kh,value_hash=vh,donor_key_hash=tensor_hash(dk),donor_value_hash=tensor_hash(dv),key_shape=list(dk.shape),value_shape=list(dv.shape),dtype=str(k.dtype)))
    return k,v
  cache=Cache();saved={}
  def forward_hook(module,args,kwargs):
   ids=kwargs.get('input_ids',args[0] if args else None);self.forward_tokens=int(ids.shape[1])
  hook=self.model.model.register_forward_pre_hook(forward_hook,with_kwargs=True)
  trace=None
  if capture and tau is not None:
   from trace_support import SuffixTrace
   trace=SuffixTrace(self.model,n,tau,4096,tensor_hash)
  enc={k:v.cuda() for k,v in enc.items()};t.cuda.reset_peak_memory_stats();start=time.time()
  try:
   with t.inference_mode():out=self.model.generate(**enc,past_key_values=cache,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,pad_token_id=self.tok.eos_token_id)
  finally:
   hook.remove()
   if trace is not None:trace.close()
  t.cuda.synchronize();pred=self.tok.decode(out[0,n:],skip_special_tokens=True);score=float(self.scorer(row['family'],pred,row['reference']));assert math.isfinite(score) and 0<=score<=1
  rec=dict(sequence=seq,score=score,prediction_hash=hashlib.sha256(pred.encode()).hexdigest(),seconds=time.time()-start,peak_allocated_bytes=t.cuda.max_memory_allocated(),peak_reserved_bytes=t.cuda.max_memory_reserved(),generation_update_count=0,kv_replaced_layers=cache.replacements)
  rec['suffix_trace']=trace.records if trace is not None else {}
  if trace is not None:assert len(trace.records)==len(self.model.model.layers)*8*(1+bool(n%4096))+2,('TRACE_COVERAGE',len(trace.records))
  rec['observed_update_calls']=len(self.write_calls)
  assert len(self.write_calls)==sum(len(MAPPING[a]) for a in seq)
  assert all(z['valid_tokens']==4095 and z['update_count']==1 and z['forward_tokens']==n for z in self.write_calls),'WRITE_SCOPE'
  if retain:
   saved['donor']={i:(z.keys[...,:plen,:].detach().cpu().clone(),z.values[...,:plen,:].detach().cpu().clone()) for i,z in enumerate(cache.layers)}
   rec['donor']={str(i):dict(prefix_tokens=plen,key_hash=tensor_hash(k),value_hash=tensor_hash(v),key_shape=list(k.shape),value_shape=list(v.shape),dtype=str(k.dtype)) for i,(k,v) in saved['donor'].items()}
  finals={};bh={};ah={};counts={}
  for l in LAYERS:
   m=self.model.model.layers[l].mlp;assert len(m._events)==len(seq);counts[str(l)]=sum(x['active_update'] for x in m._events);assert counts[str(l)]==sum(l in MAPPING[a] for a in seq)
   if capture or retain:
    finals[str(l)]=tensor_hash(m._captured_final_state)
    if tau is not None:bh[str(l)]=tensor_hash(m._mech_boundary);ah[str(l)]=tensor_hash(m._mech_after)
   if retain:saved.setdefault('weights',{})[l]=m._mech_boundary.detach().clone()
   m._events=[];m._captured_final_state=None;m._mech_boundary=None;m._mech_after=None;m._mech_inject=None
  rec.update(layer_update_counts=counts,final_hashes=finals,boundary_before_hashes=bh,boundary_after_hashes=ah)
  assert self.versions=={n:p._version for n,p in self.model.named_parameters()}
  if donor is not None:assert sorted(z['layer'] for z in cache.replacements)==list(range(len(self.model.model.layers)))
  if capture:print(json.dumps(dict(event='captured_replay',sample_id=row['sample_id'],tau=tau,inject=inject is not None,donor=donor is not None,seconds=rec['seconds'],trace_fields=len(rec['suffix_trace']))),flush=True)
  del cache,out,enc
  return rec,saved
 def mechanisms(self,x,smoke=False):
  row=getrow(x['sample_id']);seq=x['sequence'];tau=x['tau'];eligible=x['reset_screen']['eligible'];result={}
  native,keep=self.run(row,seq,tau=tau if eligible else None,retain=eligible,capture=True)
  assert native['prediction_hash']==x['prediction_hash'] and abs(native['score']-x['score'])<1e-12,('NATIVE_PARITY',x['sample_id'],native,x)
  result['native']=native;result['reverse']=self.run(row,list(reversed(seq)))[0]
  if eligible:
   reset={l:self.model.model.layers[l].mlp.down_proj.weight.detach() for l in LAYERS}
   result['reset']=self.run(row,seq,tau=tau,inject=reset,capture=True)[0]
   assert result['reset']['boundary_after_hashes']=={str(l):tensor_hash(v) for l,v in reset.items()}
   control=['OFF']*tau+seq[tau:]
   result['control']=self.run(row,control,tau=tau,capture=True)[0]
   k0,_=self.run(row,control,tau=tau,inject=keep['weights'],capture=True)
   k1,_=self.run(row,control,tau=tau,inject=keep['weights'],donor=keep['donor'],capture=True)
   assert k1['prediction_hash']==native['prediction_hash'] and k1['score']==native['score'] and k1['final_hashes']==native['final_hashes'],'KV_NATIVE_CLOSURE'
   assert native['suffix_trace'] and k1['suffix_trace']==native['suffix_trace'],'KV_SUFFIX_TRACE_CLOSURE'
   assert k0['boundary_after_hashes']==k1['boundary_after_hashes']==native['boundary_before_hashes'],'KV_WEIGHT_EQUALITY'
   result['kv']={'control_kv':k0,'native_kv':k1,'native_closure':'PASS','suffix_trace_closure':'PASS','control_trace_equals_native':k0['suffix_trace']==native['suffix_trace'],'P_factor':False}
  else:result['reset']=result['kv']={'status':'INELIGIBLE','screen':x['reset_screen']}
  del keep
  repeat=self.run(row,seq,tau=tau if eligible else None,capture=True)[0]
  assert (repeat['prediction_hash'],repeat['score'],repeat['final_hashes'],repeat['suffix_trace'])==(native['prediction_hash'],native['score'],native['final_hashes'],native['suffix_trace']),'STATE_ISOLATION'
  result['native_repeat']=repeat
  if smoke:
   self.install(original=True)
   old=self.run(row,seq,capture=True)[0];assert old['prediction_hash']==native['prediction_hash'] and old['final_hashes']==native['final_hashes'],'UNMODIFIED_V2_PARITY'
   originals={}
   probes=candidates(len(seq));probes=[probes[0],probes[len(probes)//2],probes[-1]]
   for c in probes:originals[tuple(c['sequence'])]=self.run(row,c['sequence'],capture=True)[0]
   self.install()
   for c in probes:
    new=self.run(row,c['sequence'],capture=True)[0];old=originals[tuple(c['sequence'])];assert (new['prediction_hash'],new['final_hashes'])==(old['prediction_hash'],old['final_hashes'])
   for a in ACTIONS:
    rec=self.run(row,[a]*len(seq))[0];assert abs(rec['score']-x['fixed_scores'][a])<1e-12 and rec['prediction_hash']==x['constant_hashes'][a],('CONSTANT_PARITY',a)
   result['smoke_checks']={'repeat':'PASS','original_v2':'PASS','r2_three_schedules':'PASS','fixed8':'PASS','base_unchanged':param_hash(self.model)==self.base_hash};assert result['smoke_checks']['base_unchanged']
  return result
def worker(label,gpu,slot,smoke=False):
 assert int(os.environ['CUDA_VISIBLE_DEVICES'])==gpu and gpu in GPUS
 j=verify();c=j['models'][label];root=O/label;engine=Engine(c);owned=c['partitions'][slot]
 atomic(root/f'worker{slot}_loaded.json',dict(pid=os.getpid(),physical_gpu=gpu,base_hash=engine.base_hash,time=time.time(),smoke=smoke))
 if smoke:
  cohort=[x for x in c['cohort'] if x['reset_screen']['eligible']];x=max(cohort,key=lambda x:len(set(x['sequence'])))
  result=engine.mechanisms(x,smoke=True);atomic(root/'SMOKE.json',dict(status='PASS',job_sha256=sha(S/'JOB.json'),source_sha256=sha(S/'run.py'),sample=x,results=result));return
 for sid in owned:
  x=next(x for x in c['cohort'] if x['sample_id']==sid);d=root/'samples'/sid.replace(':','_');d.mkdir(parents=True,exist_ok=True)
  assert not (d/'FINAL.json').exists()
  atomic(root/f'worker{slot}_progress.json',dict(sample=sid,phase='mechanisms',gpu=gpu,time=time.time()))
  mechanisms=engine.mechanisms(x);atomic(d/'MECHANISMS.json',mechanisms)
  row=getrow(sid);cs=candidates(len(x['sequence']));records=[];engine.torch.cuda.empty_cache()
  for i,can in enumerate(cs):
   rec=engine.run(row,can['sequence'])[0];rec={**can,**rec};records.append(rec)
   if (i+1)%8==0 or i==len(cs)-1:
    atomic(d/'R2_PROGRESS.json',dict(records=records,completed=i+1,total=len(cs)))
    atomic(root/f'worker{slot}_progress.json',dict(sample=sid,phase='r2',gpu=gpu,completed=i+1,total=len(cs),seconds=rec['seconds'],time=time.time()))
  best=max(z['score'] for z in records);tied=[z for z in records if abs(z['score']-best)<=1e-12];win=min(tied,key=lambda z:(z['tau'],ACTIONS.index(z['A']),ACTIONS.index(z['B']),z['full_schedule_hash']))
  assert param_hash(engine.model)==engine.base_hash
  atomic(d/'FINAL.json',dict(status='COMMITTED',sample_id=sid,parent_dynamic=x,mechanisms=mechanisms,r2=dict(candidate_count=len(records),best_score=best,winner=win,ties=len(tied),all_candidates_sha256=sha(d/'R2_PROGRESS.json')),base_unchanged=True,job_sha256=sha(S/'JOB.json')))
 atomic(root/f'worker{slot}_done.json',dict(status='PASS',samples=len(owned),gpu=gpu,time=time.time()))
def spawn(label,gpu,slot,smoke=False):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 log=(O/label/('smoke.log' if smoke else f'worker{slot}.log')).open('a')
 return subprocess.Popen([sys.executable,'-B',str(S/'run.py'),'worker','--model',label,'--gpu',str(gpu),'--slot',str(slot)]+(['--smoke'] if smoke else []),env=env,stdout=log,stderr=subprocess.STDOUT)
def manager(smoke=False):
 j=verify(full=True);lock=(S/('smoke.lock' if smoke else 'manager.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 locks=[]
 for g in ([1,4] if smoke else GPUS):
  p=R/'work/downstream_gpu_ownership'/f'physical_gpu{g}.lock';p.parent.mkdir(exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  free=int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=43000,('GPU_BUSY',g,free)
 if smoke:
  ps=[spawn(label,gpu,0,True) for label,gpu in [('llama8b',1),('qwen4b',4)]];codes=[p.wait() for p in ps];atomic(O/'SMOKE_STATUS.json',dict(status='PASS' if codes==[0,0] else 'FAILED',codes=codes));assert codes==[0,0];return
 for label in j['models']:
  smoke_result=read(O/label/'SMOKE.json');assert smoke_result['status']=='PASS' and smoke_result['job_sha256']==sha(S/'JOB.json') and smoke_result['source_sha256']==sha(S/'run.py')
 atomic(O/'PHASE_STATUS.json',dict(status='RUNNING',manager_pid=os.getpid(),gpus=GPUS,started=time.time(),models=list(j['models']),job_sha256=sha(S/'JOB.json')))
 for label,c in j['models'].items():
  ps=[spawn(label,g,i) for i,g in enumerate(GPUS)]
  atomic(O/label/'PHASE_STATUS.json',dict(status='RUNNING',pids=[p.pid for p in ps],gpus=GPUS,started=time.time()))
  while any(p.poll() is None for p in ps):
   if any(p.poll() not in [None,0] for p in ps):
    for p in ps:
     if p.poll() is None:p.terminate()
    break
   time.sleep(5)
  codes=[p.wait() for p in ps];atomic(O/label/'PHASE_STATUS.json',dict(status='COMPLETE' if codes==[0]*7 else 'FAILED',exit_codes=codes,ended=time.time()))
  if codes!=[0]*7:raise RuntimeError(('WORKER_FAILED',label,codes))
  results=[read(p) for p in (O/label/'samples').glob('*/FINAL.json')];assert {x['sample_id'] for x in results}=={x['sample_id'] for x in c['cohort']}
  atomic(O/label/'AGGREGATE.json',dict(samples=len(results),per_sample=results))
 atomic(O/'PHASE_STATUS.json',dict(status='COMPLETE',ended=time.time(),gpus=GPUS,job_sha256=sha(S/'JOB.json')))
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','manager','worker']);p.add_argument('--smoke',action='store_true');p.add_argument('--model');p.add_argument('--gpu',type=int);p.add_argument('--slot',type=int);a=p.parse_args()
 try:
  if a.mode=='prepare':prepare()
  elif a.mode=='manager':manager(a.smoke)
  else:worker(a.model,a.gpu,a.slot,a.smoke)
 except BaseException as e:
  atomic(O/('ERROR_'+(a.model or a.mode)+'_'+str(a.slot)+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
  if a.mode=='manager':atomic(O/('SMOKE_STATUS.json' if a.smoke else 'PHASE_STATUS.json'),dict(status='FAILED',error=repr(e),ended=time.time()))
  raise
if __name__=='__main__':main()
