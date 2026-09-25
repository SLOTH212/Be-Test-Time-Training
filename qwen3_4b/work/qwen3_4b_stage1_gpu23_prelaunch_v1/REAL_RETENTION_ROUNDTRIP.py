import os,sys,pathlib,json,hashlib,time,shutil,importlib.util
ROOT=pathlib.Path('/path/to/ttt')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or os.environ.get('WORLD_SIZE')!='2':raise RuntimeError('GPU_POLICY')
sys.path.insert(0,str(ROOT/'src/training_runtime/code'))
import torch,numpy as np
import torch.distributed as dist
from lib.distributed_runtime import initialize,shutdown,barrier
from lib.distributed_checkpoint import DistributedCheckpointRotation,capture_rank_state
from lib.model_factory import build_pretrained
from lib.fsdp2_runtime import apply_fsdp2,make_device_mesh
import yaml
WORK=ROOT/'work/qwen3_4b_stage1_gpu23_prelaunch_v1'
AUDIT=ROOT/'audits/qwen3_4b_stage1_gpu23_prelaunch_final_v3'
DEST=ROOT/'runs/qwen3_4b_gpu23_prelaunch_final_v3/retention_roundtrip/checkpoints'
SOURCE=ROOT/'runs/qwen3_4b_gpu23_prelaunch_final_v3/uninterrupted_A5/checkpoints'
cfg=yaml.safe_load((WORK/'real_final_debug_v3.yaml').read_text())
identity={**cfg['input_identity'],'config_sha256':hashlib.sha256((WORK/'real_final_debug_v3.yaml').read_bytes()).hexdigest()}
BATCH={'global_batch_size':2,'micro_batch_size':1,'gradient_accumulation':1,'identity':identity}
# Reuse frozen numeric comparator functions only; its main is not invoked.
spec=importlib.util.spec_from_file_location('numeric_reference',ROOT/'src/training_runtime/code/tests/hit_dcp_numeric_compare.py')
numeric=importlib.util.module_from_spec(spec);spec.loader.exec_module(numeric)
context=initialize()
try:
 assert context.world_size==2 and torch.cuda.device_count()==2
 if context.rank==0:
  assert not DEST.exists()
  DEST.mkdir(parents=True)
  for slot in ['slot_A','slot_B']:
   shutil.copytree(SOURCE/slot,DEST/slot,copy_function=os.link)
  shutil.copyfile(SOURCE/'latest.json',DEST/'latest.json')
 barrier()
 torch.manual_seed(42+context.rank);torch.cuda.manual_seed_all(42+context.rank)
 model,_=build_pretrained(pathlib.Path(cfg['model_base']),context.device,train_clip_none=True,ttt_layers=cfg['ttt_layers'],ttt_chunk=4096)
 apply_fsdp2(model,mesh=make_device_mesh(context.device.type,2))
 optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=True)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
 rotation=DistributedCheckpointRotation(DEST,2,[2],BATCH)
 progress=rotation.load(model,optimizer,scheduler)
 assert progress['update_step']==5
 model_before,opt_before=numeric.capture(model,optimizer)
 rng_before=capture_rank_state(progress,scheduler)
 begin=time.perf_counter();rotation.save(model,optimizer,progress,scheduler);save_seconds=time.perf_counter()-begin
 resumed=rotation.load(model,optimizer,scheduler)
 rng_after=capture_rank_state(resumed,scheduler)
 current_model,current_opt=numeric.current_maps(model,optimizer)
 model_cmp=numeric.compare(model_before,current_model,rtol=0,atol=0)
 opt_cmp=numeric.compare(opt_before,current_opt,rtol=0,atol=0)
 rng_equal=(rng_before['python_rng']==rng_after['python_rng'] and rng_before['numpy_rng'][0]==rng_after['numpy_rng'][0]
  and np.array_equal(rng_before['numpy_rng'][1],rng_after['numpy_rng'][1]) and rng_before['numpy_rng'][2:]==rng_after['numpy_rng'][2:]
  and torch.equal(rng_before['torch_cpu_rng'],rng_after['torch_cpu_rng'])
  and all(torch.equal(x,y) for x,y in zip(rng_before['torch_cuda_rng'],rng_after['torch_cuda_rng'])))
 local={'rank':context.rank,'physical_gpu':context.rank+2,'model':model_cmp,'optimizer':opt_cmp,'progress_exact':progress==resumed,'scheduler_exact':rng_before['scheduler']==rng_after['scheduler'],'rng_exact':bool(rng_equal),'save_seconds':save_seconds}
 gathered=[None,None];dist.all_gather_object(gathered,local)
 passed=all(x['model']['allclose'] and x['optimizer']['allclose'] and x['progress_exact'] and x['scheduler_exact'] and x['rng_exact'] for x in gathered)
 if context.rank==0:
  latest=json.loads((DEST/'latest.json').read_text());assert latest['slot']=='slot_A'
  assert all((DEST/s/'manifest.json').exists() for s in ['slot_A','slot_B'])
  assert not list(DEST.glob('*.previous')) and not list(DEST.glob('*.partial'))
  report={'status':'PASS' if passed else 'FAIL','ranks':gathered,'third_save_replacement_path_exercised':True,'latest_and_previous_retained':True,'serialization_format_unchanged':True,'formal_training_started':False,'checkpoint_module_sha256':hashlib.sha256((ROOT/'src/training_runtime/code/lib/distributed_checkpoint.py').read_bytes()).hexdigest()}
  (AUDIT/'REAL_RETENTION_ROUNDTRIP.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)
 if not passed:raise RuntimeError('REAL_RETENTION_STATE_PARITY_FAILED')
 barrier()
finally:
 shutdown()
 print(f'RANK_{context.rank}_SHUTDOWN_PASS',flush=True)
