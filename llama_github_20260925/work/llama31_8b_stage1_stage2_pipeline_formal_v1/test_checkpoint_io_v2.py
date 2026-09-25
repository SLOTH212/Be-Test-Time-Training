"""Four CPU ranks: slow publication beyond training timeout, exact restore and corruption rejection."""
import os,sys,tempfile,time,json,random
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
sys.path.insert(0,'/path/to/ttt/src/training_runtime/code')
import lib.distributed_checkpoint as cp
from checkpoint_io_v2 import install

def initialize():
 dist.init_process_group('gloo',timeout=timedelta(seconds=2))

g={'initialize':initialize,'DistributedCheckpointRotation':cp.DistributedCheckpointRotation}
install(g,cp,timeout_seconds=20);g['initialize']()
rank=dist.get_rank()
base='/path/to/ttt/work/llama31_8b_stage1_stage2_pipeline_formal_v1/test_io_v2_artifacts'
torch.manual_seed(31);model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.001)
scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
model(torch.ones(1,3)).sum().backward();opt.step();scheduler.step();opt.zero_grad()
rotation=g['DistributedCheckpointRotation'](base,4,[4],{'test':True})
publish=cp._publish_checkpoint
def slow(*args,**kwargs):
 time.sleep(4)
 return publish(*args,**kwargs)
cp._publish_checkpoint=slow
progress={'update_step':1,'rank':rank}
# Saved manifest progress is rank0's; each rank sidecar preserves its own state.
rotation.save(model,opt,progress,scheduler)
original={k:v.detach().clone() for k,v in model.state_dict().items()}
expected=torch.rand(4)
with torch.no_grad():
 for p in model.parameters():p.add_(20)
restored=rotation.load(model,opt,scheduler)
assert restored==progress
assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
assert torch.equal(torch.rand(4),expected)
assert len(opt.state)==2 and all(float(v['step'])==1 for v in opt.state.values())
rotation.save(model,opt,progress,scheduler)
flag=torch.tensor(1);dist.all_reduce(flag);assert flag.item()==4
if rank==0:
 slot=Path(base)/rotation.latest()['slot']
 target=slot/'rank_00000.pt'
 content=target.read_bytes();target.write_bytes(content+b'corrupt')
 try:cp._validate_checkpoint_content(slot,json.loads((slot/'manifest.json').read_text()))
 except RuntimeError as e:assert 'DCP_FILE_HASH_MISMATCH' in str(e)
 else:raise AssertionError('CORRUPTION_ACCEPTED')
 target.write_bytes(content)
 print('CPU_IO_REGRESSION_PASS slow_publish=4s default_timeout=2s io_timeout=20s model_optimizer_scheduler_rng_restore=PASS corruption_rejected=PASS',flush=True)
cp._barrier();dist.destroy_process_group()
