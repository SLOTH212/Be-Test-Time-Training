"""CPU model-only consolidation using the existing DCP model-only load planner."""
from core import *
import sys
def load_model(slot):
 import torch
 from torch.distributed.checkpoint import FileSystemReader
 from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
 from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
 require(not torch.cuda.is_initialized(),'CPU_FINALIZATION_ONLY');torch.set_num_threads(8);state={}
 _load_state_dict(state,storage_reader=FileSystemReader(Path(slot)/'dcp'),planner=_EmptyStateDictLoadPlanner(keys={'model'}),no_dist=True)
 require(set(state)=={'model'},'MODEL_ONLY_DCP_KEYS');return state['model']
def state_audit(state,c):
 import torch
 inventory=read(c['state_inventory'])['schema'];require(set(state)==set(inventory),'FINAL_MODEL_KEY_SET')
 for name,value in state.items():
  expected=inventory[name];require(list(value.shape)==expected and str(value.dtype)=='torch.bfloat16','FINAL_MODEL_SHAPE_DTYPE '+name)
  for chunk in value.reshape(-1).split(1<<20):require(bool(torch.isfinite(chunk).all()),'FINAL_NONFINITE '+name)
 require(sum(p.numel() for p in state.values())==8131072000,'LLAMA_PARAMETER_COUNT')
 require(len([n for n in state if '.ttt_' in n])==18,'SIX_LAYER_NTP_STATE')
 return {'status':'PASS','finite':True,'parameter_count':8131072000,'keys':len(state),'ttt_keys':18,'renamed_keys':False}
def finalize(stage,c):
 require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU_FINALIZATION_ONLY');source();final=Path(c[f'stage{stage}']['final_root'])
 if final.exists():validate_final(final,stage,c);return
 with lock(Path(c['state_root'])/f'FINALIZE_{stage}.lock'):
  receipt=completion(stage,c);storage(c);temp=final.parent/f'.{final.name}.partial_{time.time_ns()}';temp.mkdir();state=load_model(receipt['slot']);audit=state_audit(state,c)
  sys.path[:0]=[str(R/'src/llama31_ntp_port_v1'),str(R/'src/llama31_ntp_port_v1/source')]
  from adapters.model_family import model_classes,configure
  cfg_cls,_=model_classes('llama',inference=stage==2);cfg=cfg_cls.from_pretrained(c['base_model'],local_files_only=True);configure(cfg,model_family='llama',ttt_layers=[0,6,12,18,24,30],chunk_size=4096,inference=stage==2);cfg.to_json_file(temp/'config.json')
  if stage==1:
   import torch
   torch.save(state,temp/'model.pt');atomic(temp/'manifest.json',{'format':'gated_ntp_stage1_model_only_dcp_v1','complete':True,'world_size':4,'metadata':{'source_stage':1,'record_cursor':17015,'immutable_parent':True,'DEBUG_ONLY':False},'files':{'model.pt':sha(temp/'model.pt')}})
  else:
   from safetensors.torch import save_file
   save_file({k:v.contiguous() for k,v in state.items()},str(temp/'model.safetensors'),metadata={'format':'pt'})
   # Static consumption check uses actual inference model state shape/dtype schema.
   require(cfg.ttt_chunk==4096 and cfg.ttt_lr==1 and cfg.ttt_update_clip_norm==1e-5,'INFERENCE_CONFIG')
  del state
  for name in c['tokenizer_files']:shutil.copyfile(Path(c['base_model'])/name,temp/name)
  atomic(temp/'state_dict_audit.json',audit)
  files={p.name:sha(p) for p in temp.iterdir() if p.is_file()};(temp/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(files.items())));files['SHA256SUMS.txt']=sha(temp/'SHA256SUMS.txt')
  authority={'artifact_class':f'STAGE{stage}_FINAL','DEBUG_ONLY':False,'immutable':True,'integrity':'PASS','final_root':str(final),'source_authority_sha256':sha(SOURCE),'data_authority_sha256':sha(c[f'stage{stage}']['data_authority']),'model_revision':c['model_revision'],'port_sha256':c['port_sha256'],'base_model':c['base_model'],'config_sha256':sha(c[f'stage{stage}']['config']),'final_dcp':receipt,'files':files,'state_audit':audit,**EXPECTED[stage]}
  if stage==1:authority.update(SPLIT)
  else:
   parent=Path(c['stage1']['final_root'])/'STAGE1_FINAL_AUTHORITY.json';validate_final(parent.parent,1,c);authority.update(inference_ready='PASS',stage1_final_authority_sha256=sha(parent),qa_targets=19397,supervised_answer_tokens=97619)
  atomic(temp/f'STAGE{stage}_FINAL_AUTHORITY.json',authority)
  require(not final.exists(),'REFUSE_MISMATCHING_FINAL_OVERWRITE')
  for p in temp.iterdir():
   with p.open('rb') as f:os.fsync(f.fileno())
   p.chmod(0o444)
  require(temp.parent.resolve()==final.parent.resolve(),'ATOMIC_PUBLICATION_PARENT');temp.chmod(0o555);os.rename(temp,final);fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);validate_final(final,stage,c)
