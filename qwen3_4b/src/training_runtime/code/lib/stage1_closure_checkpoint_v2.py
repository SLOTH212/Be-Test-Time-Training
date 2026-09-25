"""Stage1 V2 DCP restore adapter preserving absent lazy optimizer state."""
import json
import torch
from torch.distributed.checkpoint import FileSystemReader
from .distributed_checkpoint import DistributedCheckpointRotation as BaseRotation
class Stage1ClosureCheckpointRotation(BaseRotation):
 def load(self,model,optimizer,scheduler=None):
  progress=super().load(model,optimizer,scheduler)
  if progress is None:return progress
  latest=json.loads((self.root/'latest.json').read_text())
  metadata=FileSystemReader(self.root/latest['slot']/'dcp').read_metadata()
  saved_names={path[2] for path in metadata.planner_data.values() if len(path)>2 and tuple(path[:2])==('optimizer','state')}
  named=dict(model.named_parameters());extra=[]
  for name,parameter in named.items():
   if parameter not in optimizer.state or name in saved_names:continue
   state=optimizer.state[parameter]
   if not name.endswith('.mlp.ttt_conv.weight'):raise RuntimeError('UNEXPECTED_OPTIMIZER_STATE_NOT_IN_CHECKPOINT '+name)
   if set(state)!={'step','exp_avg','exp_avg_sq'}:raise RuntimeError('UNEXPECTED_LAZY_STATE_SCHEMA '+name)
   for key in ['exp_avg','exp_avg_sq']:
    value=state[key].to_local() if hasattr(state[key],'to_local') else state[key]
    if bool(torch.count_nonzero(value)):raise RuntimeError('NONZERO_UNSERIALIZED_OPTIMIZER_STATE '+name)
   if float(state['step']) not in (0.0,1.0):raise RuntimeError('UNEXPECTED_LAZY_INITIALIZATION_STEP '+name)
   extra.append({'name':name,'initialization_step':float(state['step']),'moments_exact_zero':True})
   del optimizer.state[parameter]
  missing=[name for name in saved_names if name not in named or named[name] not in optimizer.state]
  if missing:raise RuntimeError('SAVED_OPTIMIZER_STATE_MISSING '+str(missing))
  self.stage1_lazy_restore_audit={'saved_state_count':len(saved_names),'removed_initialization_only_state':extra,'scientific_state_modified':False}
  print(json.dumps({'event':'STAGE1_V2_LAZY_OPTIMIZER_RESTORE',**self.stage1_lazy_restore_audit}),flush=True)
  return progress
