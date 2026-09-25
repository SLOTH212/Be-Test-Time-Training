#!/usr/bin/env python3
"""Deterministic engineering controls; execute the unchanged Stage1 worker."""
import os,sys,pathlib,json,hashlib
ROOT=pathlib.Path('/path/to/ttt')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or os.environ.get('WORLD_SIZE')!='2' or os.environ.get('LOCAL_RANK') not in ('0','1'):
    raise RuntimeError('DETERMINISTIC_WRAPPER_GPU_POLICY')
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or os.environ.get('PYTHONHASHSEED')!='42':
    raise RuntimeError('DETERMINISTIC_ENVIRONMENT_REQUIRED_BEFORE_PROCESS_START')
for key in ['TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','TMPDIR']:
    value=os.environ.get(key)
    if not value or not pathlib.Path(value).resolve().is_relative_to(ROOT):
        raise RuntimeError('DETERMINISTIC_WRAPPER_CACHE_POLICY '+key)
import yaml
config=pathlib.Path(sys.argv[sys.argv.index('--config')+1])
cfg=yaml.safe_load(config.read_text())
if cfg.get('deterministic_runtime',{}).get('wrapper_sha256')!=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest():
    raise RuntimeError('DETERMINISTIC_WRAPPER_IDENTITY')
import torch
torch.use_deterministic_algorithms(True,warn_only=False)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
print(json.dumps({'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_allow_tf32_unchanged':torch.backends.cuda.matmul.allow_tf32,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'PYTHONHASHSEED':os.environ['PYTHONHASHSEED'],'physical_gpu_allowlist':[2,3]}),flush=True)
import runpy
worker=ROOT/'src/training_runtime/code/workers/distributed_train_worker.py'
sys.argv[0]=str(worker)
runpy.run_path(str(worker),run_name='__main__')
