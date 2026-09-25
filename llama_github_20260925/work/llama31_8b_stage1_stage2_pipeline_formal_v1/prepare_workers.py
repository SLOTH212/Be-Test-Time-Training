"""Isolated architecture/data-layout adaptation; preserve existing loss blocks."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
R=Path('/path/to/ttt');W=R/'work/llama31_8b_stage1_stage2_pipeline_formal_v1';C=R/'src/training_runtime/code'
sources={1:R/'work/llama31_8b_compat_smoke_v1/worker.py',2:C/'workers/distributed_stage2_train_worker_v1.py'}
manifest={}
for stage,p in sources.items():
 original=p.read_text();s=original
 if stage==2:
  s=s.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(C)!r})').replace('!= "2,3"','!= "0,1,2,3"').replace('!= 2 or int(os.environ.get("LOCAL_RANK", "-1")) not in (0, 1)','!= 4 or int(os.environ.get("LOCAL_RANK", "-1")) not in (0, 1, 2, 3)').replace('"physical_gpu_allowlist", "2,3"','"physical_gpu_allowlist", "0,1,2,3"').replace('parameter_count != 4061881856','parameter_count != 8131072000').replace('!= 36','!= 32').replace('"layers":36','"layers":32').replace('"physical_gpu": 2 + context.local_rank','"physical_gpu": context.local_rank').replace('!= 151643','!= 128001')
 # NPZ pretokenized IDs are the frozen dataset. Never decode/re-tokenize them.
 s=s.replace('tokenizer(row["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)','torch.tensor([row["input_ids"]], dtype=torch.long, device=context.device)')
 s=s.replace('tokenizer(dummy["text"], add_special_tokens=False, return_tensors="pt").input_ids.to(context.device)','torch.tensor([dummy["input_ids"]], dtype=torch.long, device=context.device)')
 begin=s.index('        dataset_root = Path(cfg["train_data"])');end=s.index('        if args.resume_from:',begin)
 s=s[:begin]+'''        for bound_path,bound_hash in cfg['dataset_file_bindings'].items():
            if hashlib.sha256(Path(bound_path).read_bytes()).hexdigest()!=bound_hash:
                raise RuntimeError('CURRENT_DATASET_IDENTITY_MISMATCH')
'''+s[end:]
 s=s.replace('QWEN3_4B_TTT_MODEL_PARAMETER_IDENTITY','LLAMA31_8B_TTT_MODEL_PARAMETER_IDENTITY')
 s=s.replace('if args.stage == 1 and not math.isfinite(global_mean_loss):','if not math.isfinite(global_mean_loss):')
 # Existing exact core loop gets only data-reader overrides, not mathematics edits.
 insert='\nfrom data_adapter import windows, answer_ranges\n'
 loc=s.index('\nif __name__ == "__main__":');s=s[:loc]+insert+s[loc:]
 ast.parse(s);dst=W/f'worker_stage{stage}.py';dst.write_text(s)
 (W/f'worker_stage{stage}.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(dst))))
 manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(W/'worker_reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
entry='''import os,sys,runpy
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
g['main']()
'''
(W/'entry.py').write_text(entry)
print('ISOLATED_WORKERS_PREPARED_NO_LAUNCH')
