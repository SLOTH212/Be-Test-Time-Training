import sys,os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5' or os.environ.get('WORLD_SIZE')!='2':raise RuntimeError('GPU45_RESOURCE_REQUIRED')
sys.path.insert(0,'/path/to/ttt/work/qwen3_4b_stage1_stage2_pipeline_formal_v1')
from gpu45_resource_loader import execute_worker
import sys,runpy
from pathlib import Path
R=Path('/path/to/ttt')
sys.path.insert(0,str(R/'src/training_runtime/code'))
import lib.distributed_checkpoint as dc
from lib.stage2_checkpoint_runtime import Stage2CheckpointRotation
dc.DistributedCheckpointRotation=Stage2CheckpointRotation
execute_worker(str(R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'),run_name='__main__')
