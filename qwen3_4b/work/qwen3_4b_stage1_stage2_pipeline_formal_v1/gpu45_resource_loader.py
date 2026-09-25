"""Explicit user-authorized GPU index adaptation of immutable scientific source."""
import ast,hashlib,json,os
from pathlib import Path
R=Path('/path/to/ttt')
AUTH=R/'provenance/QWEN3_4B_PIPELINE_GPU45_RESOURCE_AUTHORITY_V1.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resource_contract():
 a=json.loads(AUTH.read_text())
 if sha(AUTH)!=AUTH.with_suffix('.sha256').read_text().split()[0]:raise RuntimeError('GPU45_RESOURCE_AUTHORITY_HASH')
 if a['physical_gpu_allowlist']!=[4,5] or a['world_size']!=2:raise RuntimeError('GPU45_RESOURCE_POLICY')
 for path,h in a['files'].items():
  if sha(path)!=h:raise RuntimeError('GPU45_RESOURCE_FILE_HASH '+path)
 return a
def adapt(source):
 replacements=[('"2,3"','"4,5"'),('"physical_gpu": 2 + context.local_rank','"physical_gpu": 4 + context.local_rank')]
 assert source.count('"2,3"')==2 and source.count(replacements[1][0])==1
 result=source
 for old,new in replacements:result=result.replace(old,new)
 reverse=result
 for old,new in reversed(replacements):reverse=reverse.replace(new,old)
 assert reverse==source
 return result
def execute_worker(path,run_name):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='4,5' or os.environ.get('WORLD_SIZE')!='2':raise RuntimeError('GPU45_REQUIRED_BEFORE_WORKER_IMPORT')
 resource_contract();path=Path(path);source=adapt(path.read_text());namespace={'__name__':run_name,'__file__':str(path),'__package__':None,'__spec__':None}
 exec(compile(source,str(path)+'<GPU45_RESOURCE_ONLY>','exec'),namespace);return namespace
