import os,sys,runpy,json
from datetime import timedelta
from pathlib import Path
import torch,torch.distributed as dist
R=Path('/path/to/ttt')
assert '--max-records' in sys.argv and '--debug-start-record' in sys.argv
sys.path.insert(0,str(R/'src/training_runtime/code'))
from lib.pipeline_core import load_yaml
assert not load_yaml(sys.argv[sys.argv.index('--config')+1]).get('formal_training')
orig_init=dist.init_process_group
def init(*a,**k):k['timeout']=timedelta(seconds=60);return orig_init(*a,**k)
dist.init_process_group=init
import torch.distributed.fsdp._fully_shard._fsdp_param_group as pg
orig=pg.foreach_reduce
seq=0
def trace(params,grads,*a,**k):
 global seq
 seq+=1
 print('REDUCE_TRACE '+json.dumps({'rank':int(os.environ['RANK']),'seq':seq,'params':[getattr(p,'_param_fqn','?') for p in params],'grads':[[list(g.shape),str(g.dtype)] for g in grads],'numel':sum(g.numel() for g in grads)}),flush=True)
 return orig(params,grads,*a,**k)
pg.foreach_reduce=trace
runpy.run_path(str(R/'bin/stage1_runtime_entry_v2.py'),run_name='__main__')
