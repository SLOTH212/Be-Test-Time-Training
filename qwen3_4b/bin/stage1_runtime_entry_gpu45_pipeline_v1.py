import sys,os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5' or os.environ.get('WORLD_SIZE')!='2':raise RuntimeError('GPU45_RESOURCE_REQUIRED')
sys.path.insert(0,'/path/to/ttt/work/qwen3_4b_stage1_stage2_pipeline_formal_v1')
from gpu45_resource_loader import execute_worker
#!/usr/bin/env python3
"""Stage1 V2 entry; unchanged original scientific worker plus exact lazy-state restore."""
import os,sys,hashlib,runpy
from pathlib import Path
R=Path('/path/to/ttt')
for key in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','TMPDIR'):
 value=os.environ.get(key)
 if not value or not Path(value).resolve().is_relative_to(R):raise RuntimeError('CACHE_ROOT_POLICY '+key)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5':raise RuntimeError('GPU_ALLOWLIST_REQUIRED')
P=R/'src/training_runtime/code/lib/stage1_closure_checkpoint_v2.py'
if hashlib.sha256(P.read_bytes()).hexdigest()!='56d0883f4d394db54b1c1b322830dc1a6bc360b065b5e1d402c482bd437a7605':raise RuntimeError('STAGE1_V2_ADAPTER_HASH_MISMATCH')
sys.path.insert(0,str(R/'src/training_runtime/code'))
import torch
if torch.are_deterministic_algorithms_enabled():raise RuntimeError('FORMAL_ORIGINAL_MODE_REQUIRED')
import lib.distributed_checkpoint as cp
from lib.stage1_closure_checkpoint_v2 import Stage1ClosureCheckpointRotation
cp.DistributedCheckpointRotation=Stage1ClosureCheckpointRotation
G=R/'src/training_runtime/code/lib/stage1_sparse_grad_v2.py'
if hashlib.sha256(G.read_bytes()).hexdigest()!='a1d5b2f40f46e2255de5fc2335417bc5fbd752356ab70d71b3e07f58a7dd8ff7':raise RuntimeError('STAGE1_SPARSE_GRAD_HASH_MISMATCH')
from lib.stage1_sparse_grad_v2 import install
namespace=execute_worker(str(R/'src/training_runtime/code/workers/distributed_train_worker.py'),run_name='stage1_closure_v2_worker')
install(namespace['main'].__globals__)
namespace['main']()
