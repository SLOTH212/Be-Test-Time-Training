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
rank=int(os.environ['RANK']);role=os.environ['V3_ROLE'];replica=os.environ['V3_REPLICA'];assert role in ['U','R_PRE','R_POST']
cfg=load_yaml(sys.argv[sys.argv.index('--config')+1]);assert not cfg.get('formal_training') and cfg['ttt_chunk_size']==4096 and not cfg['save_steps']
assert not torch.are_deterministic_algorithms_enabled()
out=Path(sys.argv[sys.argv.index('--output-dir')+1]).resolve();assert out.is_relative_to(R/'runs/qwen3_4b_stage1_resume_v3');out.mkdir(parents=True,exist_ok=True)
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def runtime():return {'rank':rank,'pid':os.getpid(),'replica':replica,'role':role,'protocol_sha256':os.environ['V3_PROTOCOL_SHA256'],'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'WORLD_SIZE':int(os.environ['WORLD_SIZE']),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'deterministic':torch.are_deterministic_algorithms_enabled(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'float32_matmul_precision':torch.get_float32_matmul_precision(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')}
def rng_only():return {k:normal(v) for k,v in cp.capture_rank_state({},None).items() if 'rng' in k}
def fsdp_lifecycle(model):
 from torch.distributed.fsdp import FSDPModule
 rows=[]
 for name,module in model.named_modules():
  if isinstance(module,FSDPModule):
   st=module._get_fsdp_state();pg=st._fsdp_param_group
   rows.append({'name':name,'state':str(st._training_state),'param_group_state':str(pg._training_state) if pg else None,'post_forward_order_n':len(st._comm_ctx.post_forward_order),'reduce_scatter_pending':st._comm_ctx.reduce_scatter_state is not None})
 return rows
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
  self.step=int(result['update_step']) if result else 0;self.first_window=True;self.restored_rng=None
  if result:
   got=snapshot(model,optimizer,result,scheduler);expected=json.loads((self.root/f'presave_step2_rank{rank}.json').read_text());diff=[k for k in got if got[k]!=expected[k]]
   write(out/f'load_completeness_rank{rank}.json',{'status':'PASS' if not diff else 'FAIL','differences':diff,'actual':got,'FSDP':fsdp_lifecycle(model),'runtime':runtime()})
   bad=torch.tensor(int(bool(diff)),device=rank);dist.all_reduce(bad,op=dist.ReduceOp.MAX);assert not int(bad),'FRESH_LOAD_RAW_STATE_MISMATCH'
   latest=self.latest();saved=torch.load(self.root/latest['slot']/f'rank_{rank:05d}.pt',map_location='cpu',weights_only=False);cp.restore_rank_state(saved);self.restored_rng=rng_only()
  def observe_step(opt,args,kwargs):
   self.step+=1;total=0.;numel=0
   for parameter in model.parameters():
    t=parameter.to_local() if hasattr(parameter,'to_local') else parameter
    t=t.detach().cpu().float();total+=float(t.square().sum(dtype=torch.float64));numel+=t.numel()
   with (out/f'parameter_norm_rank{rank}.jsonl').open('a') as f:f.write(json.dumps({'update_step':self.step,'local_unique_parameter_squared_norm':total,'local_numel':numel,'time':time.time()})+'\n')
  optimizer.register_step_post_hook(observe_step)
  return result
 def save(self,model,optimizer,progress,scheduler=None):
  if role=='R_PRE':
   assert progress['update_step']==2 and progress['record_cursor']==4
   before=snapshot(model,optimizer,progress,scheduler);write(self.root/f'presave_step2_rank{rank}.json',before)
   result=super().save(model,optimizer,progress,scheduler);after=snapshot(model,optimizer,progress,scheduler)
   write(out/f'boundary_save_rank{rank}.json',{'status':'PASS' if before==after else 'FAIL','raw_state_preserved':before==after,'FSDP':fsdp_lifecycle(model)})
   assert before==after,'BOUNDARY_SAVE_MUTATION';return result
  assert progress['update_step']==5 and progress['record_cursor']==10
  # Final diagnostic export only. U has no boundary save. Both U and R use
  # the identical export; no future training follows this observer operation.
  data={};meta={}
  def put(key,value):
   local=value.to_local() if hasattr(value,'to_local') else value
   data[key]=local.detach().cpu().contiguous();meta[key]={'global_shape':list(value.shape),'local_shape':list(local.shape),'dtype':str(value.dtype),'replicated':not hasattr(value,'to_local')}
  for name,value in model.state_dict().items():put('model'+'.'+name,value)
  names={id(p):n for n,p in model.named_parameters()}
  for parameter,state in optimizer.state.items():
   for name,value in state.items():
    assert isinstance(value,torch.Tensor)
    put('optimizer.state.'+names[id(parameter)]+'.'+name,value)
  assert sum(k.startswith('model.') for k in data)==417
  assert sum(k.startswith('optimizer.') for k in data)==1230
  path=out/f'final_rank{rank}.pt';tmp=path.with_suffix('.partial')
  with tmp.open('xb') as file:
   writer=HashWriter(file);torch.save(data,writer);writer.flush();os.fsync(file.fileno());digest=writer.digest.hexdigest();nbytes=writer.n
  tmp.replace(path);assert path.stat().st_size==nbytes
  write(out/f'final_rank{rank}.json',{'status':'PASS','path':str(path),'sha256':digest,'bytes':nbytes,'tensors':meta,'progress':progress,'scheduler':scheduler.state_dict(),'param_groups':[{k:([names[id(p)] for p in v] if k=='params' else normal(v)) for k,v in group.items()} for group in optimizer.param_groups],'runtime':runtime(),'FSDP':fsdp_lifecycle(model)})
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
 return original_count(window,*args,**kwargs)
g['count_window']=count_window
write(out/f'process_start_rank{rank}.json',{'time':time.time(),'runtime':runtime()})
namespace['main']()
write(out/f'process_exit_rank{rank}.json',{'status':'PASS','time':time.time(),'runtime':runtime()})
