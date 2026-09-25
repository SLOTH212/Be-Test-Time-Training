import sys,runpy
from pathlib import Path
R=Path('/path/to/ttt')
sys.path.insert(0,str(R/'src/training_runtime/code'))
import lib.distributed_checkpoint as dc
from lib.stage2_checkpoint_runtime import Stage2CheckpointRotation
dc.DistributedCheckpointRotation=Stage2CheckpointRotation
runpy.run_path(str(R/'src/training_runtime/code/workers/distributed_stage2_train_worker_v1.py'),run_name='__main__')
