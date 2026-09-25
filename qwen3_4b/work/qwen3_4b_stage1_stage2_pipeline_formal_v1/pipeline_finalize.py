"""CPU finalizers wrapping proven model-only / Stage2 HF export formats."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from pipeline_core import *
def stage1_parent_fields(meta,final,checkpoint_hash,manifest_hash):
 return {**meta,'checkpoint_path':str(final),'checkpoint_sha256':checkpoint_hash,'manifest_sha256':manifest_hash,'stage1_completed':True,'STAGE1_FINAL_INTEGRITY':'PASS','STAGE1_FINAL_STATE_DICT_AUDIT':'PASS'}
def state_audit(state,c):
 import torch
 expected=read(c['expected_state_inventory'])['keys'];require(set(state)==set(expected),'STATE_DICT_KEY_SET')
 for key,tensor in state.items():
  require(list(tensor.shape)==expected[key]['shape'],'STATE_DICT_SHAPE '+key)
  require(str(tensor.dtype)==expected[key]['dtype'],'STATE_DICT_DTYPE '+key)
  for chunk in tensor.reshape(-1).split(1<<20):require(bool(torch.isfinite(chunk).all()),'NONFINITE_FINAL_WEIGHT '+key)
 require(torch.equal(state['model.embed_tokens.weight'],state['lm_head.weight']),'TIED_EMBEDDING_MISMATCH')
 total=sum(v.numel() for v in state.values());unique=total-state['lm_head.weight'].numel()
 require(unique==4061881856 and total==4450838016,'PARAMETER_COUNT')
 ntp=[k for k in state if '.ttt_' in k];require(len(ntp)==18,'TTT_LAYER_STATE_MISSING')
 return {'status':'PASS','key_count':len(state),'keys':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()},'unique_parameter_count':unique,'state_numel_with_tied_alias':total,'tied_weights_equal':True,'ntp_keys':ntp,'no_optimizer_keys':True,'finite':True,'expected_inventory_sha256':digest(c['expected_state_inventory'])}
def load_dcp_model(slot):
 import torch
 require(not torch.cuda.is_initialized(),'CPU_FINALIZATION_REQUIRED');torch.set_num_threads(8)
 from torch.distributed.checkpoint import FileSystemReader
 from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
 from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
 state={};_load_state_dict(state,storage_reader=FileSystemReader(Path(slot)/'dcp'),planner=_EmptyStateDictLoadPlanner(keys={'model'}),no_dist=True)
 require(set(state)=={'model'},'MODEL_ONLY_RECONSTRUCTION');return state['model']
def checksums(temp):
 files={str(p.relative_to(temp)):digest(p) for p in sorted(temp.rglob('*')) if p.is_file()}
 (temp/'SHA256SUMS.txt').write_text(''.join(h+'  '+name+'\n' for name,h in files.items()))
 files['SHA256SUMS.txt']=digest(temp/'SHA256SUMS.txt');return files
def freeze_tree(temp,final):
 require(not final.exists(),'REFUSE_FINAL_OVERWRITE')
 for path in temp.rglob('*'):
  require(not path.is_symlink(),'FINAL_SYMLINK_FORBIDDEN')
  if path.is_file():
   with path.open('rb') as f:os.fsync(f.fileno())
   path.chmod(0o444)
 for path in sorted([p for p in temp.rglob('*') if p.is_dir()],reverse=True):path.chmod(0o555)
 temp.chmod(0o555);fsync_dir(temp);os.rename(temp,final);fsync_dir(final.parent)
def canonical(final,stage):
 src=final/f'STAGE{stage}_FINAL_AUTHORITY.json';dst=ROOT/f'provenance/QWEN3_4B_STAGE{stage}_FINAL_AUTHORITY_V1.json'
 if dst.exists():require(dst.read_bytes()==src.read_bytes(),'CANONICAL_FINAL_AUTHORITY_COLLISION')
 else:atomic(dst,read(src))
 # atomic() canonical formatting is identical to final authority formatting.
 require(digest(dst)==digest(src),'CANONICAL_AUTHORITY_COPY_MISMATCH');return dst
def finalize(stage,run,c):
 source_contract();preflight(c);run=owned(run);final=owned(c[f'stage{stage}']['final_root'])
 if final.exists():validate_final(final,stage,c);canonical(final,stage);return final
 with lock(Path(c['state_root'])/f'FINALIZE_STAGE{stage}.lock'):
  if final.exists():validate_final(final,stage,c);canonical(final,stage);return final
  receipt=completion(run,stage,c);storage(c)
  final.parent.mkdir(parents=True,exist_ok=True)
  attempts=owned(Path(c['state_root'])/'finalization_attempts');attempts.mkdir(parents=True,exist_ok=True)
  attempt=attempts/f'stage{stage}_{time.time_ns()}';attempt.mkdir();atomic(attempt/'RUN_FINAL_AUTHORITY.json',receipt)
  temp=attempt/'model';slot=Path(receipt['checkpoint_path'])
  if stage==1:
   import torch
   state=load_dcp_model(slot);audit=state_audit(state,c);temp.mkdir();torch.save(state,temp/'model.pt');del state
   sys.path.insert(0,str(ROOT/'src/training_runtime/code'))
   from lib.model_factory import _classes,apply_ntp_contract
   C,_=_classes();cfg=apply_ntp_contract(C.from_pretrained(c['base_model']),False,[0,6,12,18,24,30],4096);cfg.to_json_file(temp/'config.json')
   for name in c['tokenizer_files']:shutil.copyfile(Path(c['base_model'])/name,temp/name)
   meta={'model_identity':'Qwen3-4B','layers':36,'ttt_layers':[0,6,12,18,24,30],'context_length':32768,'ttt_chunk_size':4096,'config_sha256':digest(c['stage1']['config']),'tokenizer_sha256':digest(Path(c['base_model'])/'tokenizer.json'),'record_cursor':receipt['record_cursor'],'progress':receipt['progress'],'source_checkpoint':str(slot),'source_manifest_sha256':receipt['checkpoint_manifest_sha256'],'PARENT_CLASS':'STAGE1_FINAL','DEBUG_ONLY':False,'immutable_parent':True,'unique_parameter_numel':4061881856,'optimizer_policy':'Stage2 fresh initialize; no Stage1 optimizer loaded'}
   atomic(temp/'manifest.json',{'format':'gated_ntp_stage1_model_only_dcp_v1','complete':True,'world_size':2,'metadata':meta,'files':{'model.pt':digest(temp/'model.pt')}})
   atomic(temp/'state_dict_audit.json',audit)
   extra=stage1_parent_fields(meta,final,digest(temp/'model.pt'),digest(temp/'manifest.json'))
  else:
   parent=Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';validate_final(parent.parent,1,c)
   require(read(run/'resolved_config.yaml')['input_identity']['parent_authority_sha256']==digest(parent),'STAGE2_PARENT_BINDING')
   export_authority={**receipt,'STAGE2_FINAL_INTEGRITY':'PASS','stage2_completed':True,'DEBUG_ONLY':False,'stage1_final_authority':str(parent),'stage1_final_authority_sha256':digest(parent)}
   authority=attempt/'STAGE2_RUN_FINAL_AUTHORITY.json';atomic(authority,export_authority)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
   subprocess.run(['bash',c['stage2']['finalizer'],'--stage2-authority',str(authority),'--output',str(temp)],env=env,check=True)
   # Existing exporter freezes its output; this is private staging, never FINAL.
   temp.chmod(0o755)
   for path in temp.iterdir():
    if path.is_file():path.chmod(0o644)
   from safetensors.torch import load_file
   state=load_file(str(temp/'model.safetensors'),device='cpu');audit=state_audit(state,c);del state
   atomic(temp/'final_state_dict_audit.json',audit);infer=read(temp/'config.json')
   require(infer['ttt_chunk']==4096 and infer['ttt_lr']==1.0 and infer['ttt_update_clip_norm']==1e-5,'INFERENCE_CONFIG')
   # Exporter authority remains unchanged and retained. This closure adds its
   # independent static integrity authority before the FINAL atomic rename.
   extra={'stage2_completed':True,'stage1_final_authority':str(parent),'stage1_final_authority_sha256':digest(parent),'INFERENCE_READY_STATIC':'PASS','inference_config_sha256':digest(c['inference_config']),'export_authority_sha256':digest(temp/'INFERENCE_PARENT_AUTHORITY.json'),'STAGE2_FINAL_STATE_DICT_AUDIT':'PASS'}
  final_authority={'artifact_class':f'STAGE{stage}_FINAL','immutable':True,'DEBUG_ONLY':False,'final_root':str(final),f'STAGE{stage}_FINAL_INTEGRITY':'PASS','created_at':time.time(),'source_run_root':str(run),'prelaunch_authority':c[f'stage{stage}']['authority'],'prelaunch_authority_sha256':digest(c[f'stage{stage}']['authority']),'formal_config_sha256':digest(c[f'stage{stage}']['config']),'dataset_authority_sha256':c[f'stage{stage}']['dataset_authority_sha256'],'base_model_authority_sha256':c['base_model_authority_sha256'],'runtime_authority_sha256':digest(SOURCE),'final_dcp_identity':receipt,'state_dict_audit':audit,'files':checksums(temp),**c[f'stage{stage}']['expected'],**extra}
  atomic(temp/f'STAGE{stage}_FINAL_AUTHORITY.json',final_authority)
  # Validate exact Stage2 parser using the staging path without publishing a
  # false final authority: temporary in-memory relocation of checkpoint_path.
  if stage==1:
   parser=module(c['stage2']['controller'],'stage1_final_staging_parser');original_load=parser.load
   parser.load=lambda path:({**final_authority,'checkpoint_path':str(temp)} if Path(path)==temp/'STAGE1_FINAL_AUTHORITY.json' else original_load(path))
   parser.parent_contract(temp/'STAGE1_FINAL_AUTHORITY.json')
  freeze_tree(temp,final);validate_final(final,stage,c);canonical(final,stage);return final
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[1,2],required=True);ap.add_argument('--run-root',required=True);args=ap.parse_args()
 require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU_FINALIZER_REQUIRED');print(finalize(args.stage,args.run_root,config()))
if __name__=='__main__':main()
