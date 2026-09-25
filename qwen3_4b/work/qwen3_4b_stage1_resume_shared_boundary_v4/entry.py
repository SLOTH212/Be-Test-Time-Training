import os,sys,json,hashlib,time,runpy
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_v3';A=R/'audits/v3'
protocol=A/'RESUME_NUMERICAL_EQUIVALENCE_PROTOCOL_V3.json'
assert hashlib.sha256(protocol.read_bytes()).hexdigest()==os.environ['V3_PROTOCOL_SHA256']
assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3' and os.environ['WORLD_SIZE']=='2'
for key in ['TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','TMPDIR']:assert Path(os.environ[key]).resolve().is_relative_to(R)
for p,h in json.loads((R/'audits/STAGE1_CLOSURE_RUNTIME_SOURCE_MANIFEST_V2.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
sys.path.insert(0,str(R/'src/training_runtime/code'));sys.path.insert(0,str(W))
import numpy as np,torch,torch.distributed as dist
import lib.distributed_checkpoint as cp
from lib.stage1_closure_checkpoint_v2 import Stage1ClosureCheckpointRotation
from lib.stage1_sparse_grad_v2 import install
from lib.pipeline_core import load_yaml
from raw_state_fingerprint import normal,snapshot
v4_audit=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4'
v4_protocol=v4_audit/'SHARED_BOUNDARY_CAUSAL_PROTOCOL_V4.json'
assert hashlib.sha256(v4_protocol.read_bytes()).hexdigest()==os.environ['V4_PROTOCOL_SHA256']
rank=int(os.environ['RANK']);role=os.environ['V3_ROLE'];replica=os.environ['V3_REPLICA'];assert role in ['U','R_PRE','R_POST']
cfg=load_yaml(sys.argv[sys.argv.index('--config')+1]);assert not cfg.get('formal_training') and cfg['ttt_chunk_size']==4096 and not cfg['save_steps']
assert not torch.are_deterministic_algorithms_enabled()
localization_plan=json.loads((A/'localization_final/CONDITIONAL_LOCALIZATION_PLAN.json').read_text())
assert hashlib.sha256((A/'localization_final/CONDITIONAL_LOCALIZATION_PLAN.json').read_bytes()).hexdigest()==os.environ['V3_LOCALIZATION_PLAN_SHA256']
selected=set(localization_plan['parameters']);captured={};capture_meta={};capture_events=[];forward_trace=[];forward_active=False;clip_observation={};observer_read_checks=0
out=Path(sys.argv[sys.argv.index('--output-dir')+1]).resolve();assert out.is_relative_to(R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4');out.mkdir(parents=True,exist_ok=True)
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def runtime():return {'rank':rank,'pid':os.getpid(),'replica':replica,'role':role,'DEBUG_ONLY':True,'v4_branch':os.environ['V4_BRANCH'],'v4_protocol_sha256':os.environ['V4_PROTOCOL_SHA256'],'protocol_sha256':os.environ['V3_PROTOCOL_SHA256'],'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'WORLD_SIZE':int(os.environ['WORLD_SIZE']),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'deterministic':torch.are_deterministic_algorithms_enabled(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'float32_matmul_precision':torch.get_float32_matmul_precision(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')}
def rng_only():return {k:normal(v) for k,v in cp.capture_rank_state({},None).items() if 'rng' in k}
def fsdp_lifecycle(model):
 from torch.distributed.fsdp import FSDPModule
 rows=[]
 for name,module in model.named_modules():
  if isinstance(module,FSDPModule):
   st=module._get_fsdp_state();pg=st._fsdp_param_group
   rows.append({'name':name,'state':str(st._training_state),'param_group_state':str(pg._training_state) if pg else None,'comm_lazy_initialized':hasattr(st._comm_ctx,'post_forward_order'),'post_forward_order_n':len(st._comm_ctx.post_forward_order) if hasattr(st._comm_ctx,'post_forward_order') else None,'reduce_scatter_pending':st._comm_ctx.reduce_scatter_state is not None if hasattr(st._comm_ctx,'reduce_scatter_state') else None})
 return rows
def capture_value(key,value,source_kind):
 assert key not in capture_meta,('DUPLICATE_CAPTURE',key)
 if value is None:
  capture_meta[key]={'present':False,'source_kind':source_kind};return
 global observer_read_checks
 before_rng=rng_only();local=value.to_local() if hasattr(value,'to_local') else value;version=local._version
 captured[key]=local.detach().cpu().contiguous().clone()
 assert local._version==version and rng_only()==before_rng,'OBSERVER_MUTATED_STATE'
 observer_read_checks+=1
 capture_meta[key]={'present':True,'source_kind':source_kind,'global_shape':list(value.shape),'local_shape':list(local.shape),'dtype':str(value.dtype),'replicated':not hasattr(value,'to_local')}
def capture_state(stage,parameters=False,moments=False,gradients=False):
 assert live is not None
 found=set()
 for name,p in live.model.named_parameters():
  if name not in selected:continue
  found.add(name)
  if parameters:capture_value(stage+'.parameter.'+name,p,'sharded_parameter')
  if gradients:capture_value(stage+'.gradient.'+name,p.grad,'sharded_gradient')
  if moments:
   for state_name in ['exp_avg','exp_avg_sq','step']:capture_value(stage+'.optimizer.'+name+'.'+state_name,live.optimizer.state.get(p,{}).get(state_name),'optimizer_state')
 assert found==selected,('LOCALIZATION_PARAMETER_MISSING',selected-found)
 capture_events.append({'stage':stage,'time':time.time(),'step_before_or_after':live.step,'rng':rng_only()})
def tensor_fingerprint(value):
 global observer_read_checks
 local=value.to_local() if hasattr(value,'to_local') else value
 version=local._version;before_rng=rng_only();v=local.detach().cpu().contiguous();flat=v.reshape(-1)
 n=flat.numel();total=0.;squared=0.;maxabs=0.;finite=0
 for start in range(0,n,1<<20):
  x=flat[start:start+(1<<20)].float().numpy().astype(np.float64)
  total+=float(x.sum());squared+=float(np.dot(x,x));maxabs=max(maxabs,float(np.abs(x).max(initial=0)));finite+=int(np.isfinite(x).sum())
 indices=np.linspace(0,max(0,n-1),min(33,n),dtype=np.int64)
 result={'shape':list(v.shape),'dtype':str(v.dtype),'numel':n,'norm':squared**.5,'max_abs':maxabs,'mean':total/max(1,n),'finite_count':finite,'sample_indices':indices.tolist(),'samples':flat[indices.tolist()].float().tolist(),'sha256':hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
 assert local._version==version and rng_only()==before_rng,'OBSERVER_MUTATED_FORWARD_STATE'
 observer_read_checks+=1;return result
def install_forward_hooks(model):
 modules=dict(model.named_modules());wanted=set(x['module'] for x in localization_plan['targets']);counts={}
 def tensors(x):
  if isinstance(x,torch.Tensor):return [x]
  if isinstance(x,(tuple,list)):return [v for y in x for v in tensors(y)]
  if isinstance(x,dict):return [v for y in x.values() for v in tensors(y)]
  return []
 def make_hook(name):
  def hook(module,args,output):
   if not forward_active:return
   counts[name]=counts.get(name,0)+1
   forward_trace.append({'ordinal':len(forward_trace),'module':name,'call':counts[name],'outputs':[tensor_fingerprint(v) for v in tensors(output)]})
  return hook
 for name in sorted(wanted):
  assert name in modules,('FORWARD_MODULE_NOT_FOUND',name)
  modules[name].register_forward_hook(make_hook(name))

def whole_state_light(model,optimizer,progress,scheduler):
 names={id(p):n for n,p in model.named_parameters()}
 def fp(v):
  if isinstance(v,torch.Tensor):
   return dict(tensor_fingerprint(v),global_shape=list(v.shape),placement=str(getattr(v,'placements',None)))
  return normal(v)
 return {'DEBUG_ONLY':True,'model':{n:fp(p) for n,p in model.named_parameters()},'buffers':{n:fp(p) for n,p in model.named_buffers()},'optimizer':{names[id(p)]:{k:fp(v) for k,v in state.items()} for p,state in optimizer.state.items()},'rank_state':normal(cp.capture_rank_state(progress,scheduler)),'runtime':runtime()}

class HashWriter:
 def __init__(self,f):self.f=f;self.digest=hashlib.sha256();self.n=0
 def write(self,b):self.digest.update(b);self.n+=len(b);return self.f.write(b)
 def flush(self):return self.f.flush()
 def tell(self):return self.f.tell()
class AuditRotation(Stage1ClosureCheckpointRotation):
 def load(self,model,optimizer,scheduler=None):
  global live
  self.model=model;self.optimizer=optimizer;self.scheduler=scheduler;live=self
  initial=snapshot(model,optimizer,{},scheduler);expected=json.loads((R/f'runs/qwen3_4b_stage1_closure_v2/U1/initial_rank{rank}.json').read_text());diff=[k for k in initial if initial[k]!=expected[k]]
  write(out/f'initial_rank{rank}.json',initial);write(out/f'initial_parent_check_rank{rank}.json',{'status':'PASS' if not diff else 'FAIL','differences':diff,'runtime':runtime()})
  bad=torch.tensor(int(bool(diff)),device=rank);dist.all_reduce(bad,op=dist.ReduceOp.MAX);assert not int(bad),'INITIAL_PARENT_VALUE_MISMATCH'
  result=super().load(model,optimizer,scheduler)
  self.loaded_progress=result
  self.step=int(result['update_step']) if result else 0;self.first_window=True;self.restored_rng=None
  if result:
   got=snapshot(model,optimizer,result,scheduler);expected=json.loads((self.root/f'presave_step2_rank{rank}.json').read_text());diff=[k for k in got if got[k]!=expected[k]]
   write(out/f'load_completeness_rank{rank}.json',{'status':'PASS' if not diff else 'FAIL','differences':diff,'actual':got,'FSDP':fsdp_lifecycle(model),'runtime':runtime()})
   bad=torch.tensor(int(bool(diff)),device=rank);dist.all_reduce(bad,op=dist.ReduceOp.MAX);assert not int(bad),'FRESH_LOAD_RAW_STATE_MISMATCH'
   self.restored_rng=rng_only()
  def observe_step(opt,args,kwargs):
   self.step+=1;total=0.;numel=0
   if self.step==3:capture_state('post_step',parameters=True,moments=True)
   for parameter in model.parameters():
    t=parameter.to_local() if hasattr(parameter,'to_local') else parameter
    t=t.detach().cpu().float();total+=float(t.square().sum(dtype=torch.float64));numel+=t.numel()
   with (out/f'parameter_norm_rank{rank}.jsonl').open('a') as f:f.write(json.dumps({'update_step':self.step,'local_unique_parameter_squared_norm':total,'local_numel':numel,'time':time.time()})+'\n')
  optimizer.register_step_post_hook(observe_step)
  install_forward_hooks(model)
  return result
 def save(self,model,optimizer,progress,scheduler=None):
  if role=='U' and progress['update_step']==2:
   assert progress['update_step']==2 and progress['record_cursor']==4
   before=snapshot(model,optimizer,progress,scheduler);write(self.root/f'presave_step2_rank{rank}.json',before)
   result=super().save(model,optimizer,progress,scheduler);after=snapshot(model,optimizer,progress,scheduler)
   write(out/f'boundary_save_rank{rank}.json',{'status':'PASS' if before==after else 'FAIL','raw_state_preserved':before==after,'FSDP':fsdp_lifecycle(model)})
   assert before==after,'BOUNDARY_SAVE_MUTATION'
   write(out/f'boundary_post_save_rank{rank}.json',after)
   return result
  assert progress['update_step']==3 and progress['record_cursor']==6
  # Final diagnostic export only. U has no boundary save. Both U and R use
  # the identical export; no future training follows this observer operation.
  names={id(p):n for n,p in model.named_parameters()}
  for name in selected:
   key='pre_reduce.gradient.'+name
   if key not in capture_meta:capture_value(key,None,'not_observed_before_reduce')
  data=captured;meta=capture_meta
  assert {x['stage'] for x in capture_events}=={'pre_forward','pre_clip','post_clip','post_step'}
  path=out/f'final_rank{rank}.pt';tmp=path.with_suffix('.partial')
  with tmp.open('xb') as file:
   writer=HashWriter(file);torch.save(data,writer);writer.flush();os.fsync(file.fileno());digest=writer.digest.hexdigest();nbytes=writer.n
  tmp.replace(path);assert path.stat().st_size==nbytes
  write(out/f'final_rank{rank}.json',{'status':'PASS','path':str(path),'sha256':digest,'bytes':nbytes,'tensors':meta,'capture_events':capture_events,'forward_trace':forward_trace,'clip_observation':clip_observation,'observer_read_checks':observer_read_checks,'selected_parameters':sorted(selected),'localization_plan_sha256':os.environ['V3_LOCALIZATION_PLAN_SHA256'],'progress':progress,'scheduler':scheduler.state_dict(),'param_groups':[{k:([names[id(p)] for p in v] if k=='params' else normal(v)) for k,v in group.items()} for group in optimizer.param_groups],'runtime':runtime(),'FSDP':fsdp_lifecycle(model)})
  write(out/f'whole_state_light_rank{rank}.json',whole_state_light(model,optimizer,progress,scheduler))
  dist.barrier();return {'diagnostic_final_export':True}
live=None
cp.DistributedCheckpointRotation=AuditRotation
namespace=runpy.run_path(str(R/'src/training_runtime/code/workers/distributed_train_worker.py'),run_name='stage1_v3_worker');g=namespace['main'].__globals__;install(g);original_count=g['count_window']
def count_window(window,*args,**kwargs):
 assert live is not None
 current=rng_only();same=(not live.first_window or live.restored_rng is None or current==live.restored_rng)
 write(out/f'preforward_step{live.step+1}_rank{rank}.json',{'rng':current,'post_load_rng_preserved':same,'cursor':window[0][0],'sample_ids':[row['sample_id'] for _,row in window],'runtime':runtime()})
 assert same,'POST_LOAD_CONSTRUCTION_CHANGED_RNG'
 live.first_window=False
 if live.step==2:
  import inspect
  progress=live.loaded_progress if role=='R_POST' else inspect.currentframe().f_back.f_locals['progress']
  if role=='U':live.save(live.model,live.optimizer,progress,live.scheduler)
  boundary=snapshot(live.model,live.optimizer,progress,live.scheduler)
  write(out/f'boundary_contract_rank{rank}.json',boundary)
  expected=json.loads((live.root/f'presave_step2_rank{rank}.json').read_text());assert boundary==expected,'SHARED_BOUNDARY_PREFORWARD_STATE_CHANGED'
  capture_state('pre_forward',parameters=True,moments=True)
 return original_count(window,*args,**kwargs)
g['count_window']=count_window
original_clip=g['clip_grad_norm_fsdp2']
def observed_clip(*args,**kwargs):
 targeted=live is not None and live.step==2
 if targeted:
  capture_state('pre_clip',gradients=True)
  for name in selected:
   key='pre_clip.gradient.'+name;alias='post_reduce.gradient.'+name;capture_meta[alias]=dict(capture_meta[key],alias_of=key)
   if key in captured:captured[alias]=captured[key]
 result=original_clip(*args,**kwargs)
 if targeted:capture_state('post_clip',gradients=True)
 return result
g['clip_grad_norm_fsdp2']=observed_clip
# Read-only observation after FSDP has collected/materialized local gradients,
# immediately before its unchanged foreach_reduce implementation.
import torch.distributed.fsdp._fully_shard._fsdp_param_group as pg_module
original_foreach_reduce=pg_module.foreach_reduce
def observed_foreach_reduce(fsdp_params,unsharded_grads,*args,**kwargs):
 if live is not None and live.step==2:
  for p,grad in zip(fsdp_params,unsharded_grads):
   name=getattr(p,'_param_fqn','')
   if name in selected:capture_value('pre_reduce.gradient.'+name,grad,'unsharded_local_gradient_before_collective')
 return original_foreach_reduce(fsdp_params,unsharded_grads,*args,**kwargs)
pg_module.foreach_reduce=observed_foreach_reduce


original_fused=g['fused_sum']
def observed_fused(*args,**kwargs):
 global forward_active
 forward_active=live is not None and live.step==2
 try:
  result=original_fused(*args,**kwargs)
  if forward_active:forward_trace.append({'ordinal':len(forward_trace),'module':'__global_fused_numerator__','call':1,'outputs':[tensor_fingerprint(result)]})
  return result
 finally:forward_active=False
g['fused_sum']=observed_fused
# The scientific clipping function is the original source with exactly one
# read-only call inserted after its actual coefficient is computed.
import importlib,ast
clip_module=importlib.import_module('torch.nn.utils.clip_grad')
import inspect
clip_source=inspect.getsource(clip_module._clip_grads_with_norm_)
needle='    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)'
assert clip_source.count(needle)==1
observed_clip_source=clip_source.replace(needle,needle+'\n    _v3_record_clip(total_norm, max_norm, clip_coef, clip_coef_clamped)')
def record_clip(norm,threshold,coefficient,clamped):
 if live is None or live.step!=2:return
 assert not clip_observation
 clip_observation.update({'threshold':float(threshold),'global_norm':tensor_fingerprint(norm),'coefficient':tensor_fingerprint(coefficient),'actual_clamped_coefficient':tensor_fingerprint(clamped)})
clip_globals=dict(clip_module.__dict__);clip_globals['_v3_record_clip']=record_clip
exec(compile(observed_clip_source,'<v3_read_only_clip_observer>','exec'),clip_globals)
clip_module._clip_grads_with_norm_=clip_globals['_clip_grads_with_norm_']

write(out/f'process_start_rank{rank}.json',{'time':time.time(),'runtime':runtime()})
namespace['main']()
write(out/f'process_exit_rank{rank}.json',{'status':'PASS','time':time.time(),'runtime':runtime()})
