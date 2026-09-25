#!/usr/bin/env python3
import argparse,json,hashlib,os,sys
from pathlib import Path
R=Path('/path/to/ttt')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def owned(p):
 p=Path(p).resolve()
 if not p.is_relative_to(R.resolve()):raise RuntimeError('OUTPUT_OR_CHECKPOINT_OUTSIDE_AUTHORIZED_ROOT')
 return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage2-authority',required=True);ap.add_argument('--output',required=True);ap.add_argument('--debug-test',action='store_true');args=ap.parse_args()
 authority=owned(args.stage2_authority);a=json.loads(authority.read_text())
 if args.debug_test:
  if a.get('DEBUG_ONLY') is not True or a.get('STAGE2_DEBUG_CHECKPOINT_INTEGRITY')!='PASS':raise RuntimeError('DEBUG_STAGE2_AUTHORITY_REQUIRED')
 else:
  if a.get('DEBUG_ONLY') or a.get('PARENT_CLASS')=='DEBUG_ONLY':raise RuntimeError('DEBUG_STAGE2_FORBIDDEN_FORMAL')
  if a.get('STAGE2_FINAL_INTEGRITY')!='PASS' or a.get('stage2_completed') is not True:raise RuntimeError('STAGE2_FINAL_REQUIRED')
  if a.get('record_cursor')!=925 or a.get('cumulative_tokens')!=30000000:raise RuntimeError('STAGE2_FINAL_COUNTERS')
 slot=owned(a['checkpoint_path']);m=json.loads((slot/'manifest.json').read_text())
 if not m.get('complete') or m['progress'].get('stage')!=2 or sha(slot/'manifest.json')!=a['checkpoint_manifest_sha256']:raise RuntimeError('STAGE2_DCP_AUTHORITY_MISMATCH')
 if not args.debug_test and (m['progress'].get('record_cursor')!=925 or m['progress'].get('cumulative_tokens')!=30000000 or m['progress'].get('cumulative_answer_positions')!=388806 or m['progress'].get('cumulative_context_positions')!=29610269):raise RuntimeError('ACTUAL_STAGE2_DCP_NOT_FINAL')
 for name,h in m['files'].items():
  f=(slot/name).resolve()
  if not f.is_relative_to(slot) or sha(f)!=h:raise RuntimeError('STAGE2_DCP_FILE_HASH_MISMATCH')
 out=owned(args.output)
 if out.exists():raise RuntimeError('REFUSE_EXPORT_OVERWRITE')
 partial=out.with_name(out.name+'.partial')
 if partial.exists():raise RuntimeError('PARTIAL_EXPORT_ALREADY_EXISTS')
 partial.mkdir(parents=True)
 import torch
 torch.set_num_threads(8)
 from torch.distributed.checkpoint import FileSystemReader
 from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
 from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
 from safetensors.torch import save_file
 sd={};_load_state_dict(sd,storage_reader=FileSystemReader(slot/'dcp'),planner=_EmptyStateDictLoadPlanner(keys={'model'}),no_dist=True)
 state=sd['model']
 sys.path.insert(0,str(R/'src/training_runtime/code'))
 from lib.model_factory import _classes,apply_ntp_contract
 C,_=_classes();cfg=apply_ntp_contract(C.from_pretrained(R/'shared/models/Qwen3-4B-Base'),False,[0,6,12,18,24,30],4096);cfg.use_cache=True
 inference_reference=R/'configs/qwen3_4b_stage2_inference_4k_v1.json'
 frozen_infer=json.loads(inference_reference.read_text())
 assert frozen_infer['ttt_chunk_size']==4096 and frozen_infer['ttt_lr']==1.0 and frozen_infer['delta_clip_frobenius']==1e-5
 for name,h in frozen_infer['runtime_file_sha256'].items():
  if sha(Path(frozen_infer['runtime_code_root'])/name)!=h:raise RuntimeError('INFERENCE_4K_RUNTIME_HASH_MISMATCH')
 cfg.ttt_chunk=frozen_infer['ttt_chunk_size'];cfg.ttt_lr=frozen_infer['ttt_lr'];cfg.ttt_update_clip_norm=frozen_infer['delta_clip_frobenius']
 cfg.to_json_file(partial/'config.json')
 from transformers import AutoTokenizer
 AutoTokenizer.from_pretrained(R/'shared/models/Qwen3-4B-Base',local_files_only=True).save_pretrained(partial)
 # DCP reconstruction materializes aliases separately; clone only a repeated storage.
 pointers=set();safe={}
 for k,v in state.items():
  v=v.contiguous();ptr=v.untyped_storage().data_ptr()
  if ptr in pointers:v=v.clone()
  pointers.add(v.untyped_storage().data_ptr());safe[k]=v
 save_file(safe,str(partial/'model.safetensors'),metadata={'format':'pt'})
 audit={'status':'PASS_DCP_EXPORT_KEY_SHAPE_INVENTORY','keys':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()},'ntp_keys':[k for k in state if '.ttt_' in k],'tied_weights_equal':torch.equal(state['model.embed_tokens.weight'],state['lm_head.weight']),'source_manifest_sha256':a['checkpoint_manifest_sha256'],'DEBUG_ONLY':args.debug_test}
 (partial/'state_dict_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 files={p.name:sha(p) for p in partial.iterdir() if p.is_file()}
 result={'status':'PASS_EXPORT_PENDING_INFERENCE_LOAD','DEBUG_ONLY':args.debug_test,'PAPER_RESULT':False if args.debug_test else None,'source_stage2_authority':str(authority),'source_stage2_authority_sha256':sha(authority),'source_checkpoint_manifest_sha256':a['checkpoint_manifest_sha256'],'model_identity':'Qwen3-4B','ttt_layers':[0,6,12,18,24,30],'training_ttt_chunk_size':4096,'inference_ttt_chunk_size':4096,'inference_reference':str(inference_reference),'inference_reference_sha256':sha(inference_reference),'context_length':32768,'files':files}
 (partial/'INFERENCE_PARENT_AUTHORITY.json').write_text(json.dumps(result,indent=2)+'\n')
 os.rename(partial,out)
 for p in out.iterdir():
  if p.is_file():p.chmod(0o444)
 out.chmod(0o555)
 print(json.dumps({'status':'EXPORTED','output':str(out),'authority_sha256':sha(out/'INFERENCE_PARENT_AUTHORITY.json'),'DEBUG_ONLY':args.debug_test}),flush=True)
if __name__=='__main__':main()
