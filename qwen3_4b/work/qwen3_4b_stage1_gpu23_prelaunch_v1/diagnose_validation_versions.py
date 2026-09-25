import os,sys,pathlib,json,hashlib,importlib.util,time
R=pathlib.Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_gpu23_prelaunch_v1';A=R/'audits/qwen3_4b_stage1_gpu23_prelaunch_corrected_v2'
assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3' and os.environ['WORLD_SIZE']=='2'
sys.path.insert(0,str(R/'src/training_runtime/code'))
import torch,yaml
import torch.distributed as dist
from lib.distributed_runtime import initialize,shutdown,barrier
from lib.distributed_checkpoint import DistributedCheckpointRotation
from lib.model_factory import build_pretrained
from lib.fsdp2_runtime import apply_fsdp2,make_device_mesh
from transformers import AutoTokenizer
from workers.distributed_train_worker import validate_stage1
cfg=yaml.safe_load((W/'real_debug_v2.yaml').read_text())
identity={**cfg['input_identity'],'config_sha256':hashlib.sha256((W/'real_debug_v2.yaml').read_bytes()).hexdigest()}
batch={'global_batch_size':2,'micro_batch_size':1,'gradient_accumulation':1,'identity':identity}
c=initialize()
try:
 torch.manual_seed(42+c.rank);torch.cuda.manual_seed_all(42+c.rank)
 model,_=build_pretrained(pathlib.Path(cfg['model_base']),c.device,train_clip_none=True,ttt_layers=cfg['ttt_layers'],ttt_chunk=4096)
 apply_fsdp2(model,mesh=make_device_mesh(c.device.type,2))
 opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=True)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 progress=DistributedCheckpointRotation(R/'runs/qwen3_4b_gpu23_prelaunch_corrected_v2/resume_B/checkpoints',2,[2],batch).load(model,opt,sched)
 tokenizer=AutoTokenizer.from_pretrained(cfg['model_base'],local_files_only=True)
 def snapshot():
  return {n:p.to_local().detach().cpu().clone() if hasattr(p,'to_local') else p.detach().cpu().clone() for n,p in model.named_parameters()}
 before=snapshot();versions={n:p._version for n,p in model.named_parameters()}
 error=None;validation=None
 try:validation=validate_stage1(model,tokenizer,cfg['validation_data'],4,c)
 except RuntimeError as e:
  if str(e)!='VALIDATION_MUTATED_WEIGHTS':raise
  error=str(e)
 after=snapshot();changes=[{'name':n,'before_version':versions[n],'after_version':p._version} for n,p in model.named_parameters() if versions[n]!=p._version]
 different=[n for n in before if not torch.equal(before[n],after[n])]
 local={'rank':c.rank,'physical_gpu':c.rank+2,'loaded_step':progress['update_step'],'guard_error':error,'version_changes':changes,'model_tensor_count':len(before),'different_model_tensors':different,'all_model_shard_bytes_equal':not different,'validation':validation}
 results=[None,None];dist.all_gather_object(results,local)
 if c.rank==0:
  report={'status':'DIAGNOSTIC_COMPLETE','ranks':results,'formal_training_started':False}
  (A/'VALIDATION_VERSION_COUNTER_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 barrier()
finally:shutdown()
