"""Public torchrun entry; installs the source-proven stage-specific adapters."""
import argparse, importlib, sys

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--family',choices=['qwen','llama'],required=True)
 parser.add_argument('--stage',type=int,choices=[1,2],required=True)
 parser.add_argument('--config',required=True)
 args,rest=parser.parse_known_args()
 from dynamic_ttt.training.lib import distributed_checkpoint as cp
 if args.stage==1:
  from dynamic_ttt.training.lib.stage1_closure_checkpoint_v2 import Stage1ClosureCheckpointRotation
  cp.DistributedCheckpointRotation=Stage1ClosureCheckpointRotation
 else:
  from dynamic_ttt.training.lib.stage2_checkpoint_runtime import Stage2CheckpointRotation
  cp.DistributedCheckpointRotation=Stage2CheckpointRotation
 worker=importlib.import_module(f'dynamic_ttt.training.{args.family}_stage{args.stage}')
 if args.stage==1:
  from dynamic_ttt.training.lib.stage1_sparse_grad_v2 import install
  install(worker.main.__globals__)
 if args.family=='llama':
  from dynamic_ttt.models.family import build
  def builder(path,device,train_clip_none=True,ttt_layers=None,ttt_chunk=4096):
   return build(path,device=device,model_family='llama',ttt_layers=ttt_layers,chunk_size=ttt_chunk)
  worker.build_pretrained=builder
  from dynamic_ttt.training.checkpoint_io import install
  install(worker.main.__globals__,cp)
 sys.argv=[sys.argv[0],'--stage',str(args.stage),'--config',args.config]+rest
 worker.main()

if __name__=='__main__':main()
