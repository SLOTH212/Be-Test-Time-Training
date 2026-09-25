import os,sys,runpy
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/llama31_8b_stage1_stage2_pipeline_formal_v1';C=R/'src/training_runtime/code'
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3' and int(os.environ['WORLD_SIZE'])==4
assert int(os.environ['LOCAL_RANK']) in range(4)
sys.path[:0]=[str(W),str(C),str(R/'src/llama31_ntp_port_v1'),str(R/'src/llama31_ntp_port_v1/source')]
stage=int(sys.argv[sys.argv.index('--stage')+1])
import lib.distributed_checkpoint as cp
if stage==1:
 from lib.stage1_closure_checkpoint_v2 import Stage1ClosureCheckpointRotation
 cp.DistributedCheckpointRotation=Stage1ClosureCheckpointRotation
else:
 from lib.stage2_checkpoint_runtime import Stage2CheckpointRotation
 cp.DistributedCheckpointRotation=Stage2CheckpointRotation
ns=runpy.run_path(str(W/f'worker_stage{stage}.py'),run_name='llama_formal_worker');g=ns['main'].__globals__
if stage==1:
 from lib.stage1_sparse_grad_v2 import install
 install(g)
from adapters.model_family import build
def builder(path,device,train_clip_none=True,ttt_layers=None,ttt_chunk=4096):
 return build(path,device=device,model_family='llama',ttt_layers=ttt_layers,chunk_size=ttt_chunk)
g['build_pretrained']=builder
from checkpoint_io_v2 import install as install_checkpoint_io
install_checkpoint_io(g, cp)
g['main']()
