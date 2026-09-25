"""Explicit manual cleanup, never called by supervisor."""
import argparse
from pipeline_core import *
def candidates(c):
 state=read(Path(c['state_root'])/'PIPELINE_STATE.json');require(state['state']=='PIPELINE_COMPLETE','PIPELINE_NOT_COMPLETE')
 for stage in (1,2):validate_final(c[f'stage{stage}']['final_root'],stage,c)
 paths=[]
 for stage in (1,2):
  run=progress(stage,c)['run_root'];require(run and not matching_processes(run),'TRAINING_STILL_ACTIVE')
  p=owned(Path(run)/'checkpoints');require(p.name=='checkpoints' and p.parent==owned(run),'INVALID_PRUNE_TARGET')
  require(all(not p.is_relative_to(Path(c[f'stage{s}']['final_root'])) and not Path(c[f'stage{s}']['final_root']).is_relative_to(p) for s in (1,2)),'FINAL_MODEL_DELETE_FORBIDDEN')
  paths.append(p)
 return paths
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args();c=config();source_contract()
 with lock(Path(c['state_root'])/'PIPELINE.lock'):
  paths=candidates(c);print(json.dumps({'mode':'EXECUTE' if args.execute else 'dry-run','exact_checkpoint_targets':[str(p) for p in paths],'final_models_logs_authorities_preserved':True},indent=2))
  if args.execute:
   for path in paths:
    require(not path.is_symlink() and path==owned(path),'PRUNE_PATH_CHANGED')
    if path.exists():shutil.rmtree(path)
   atomic(Path(c['state_root'])/'EXPLICIT_CHECKPOINT_PRUNE_RECEIPT.json',{'time':time.time(),'paths':[str(p) for p in paths],'explicit_execute':True})
if __name__=='__main__':main()
