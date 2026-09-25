"""CPU-only validation using immutable DEBUG evidence; creates no FINAL model."""
import gc,json,os,sys,time
from pathlib import Path
import torch
from pipeline_core import config,ROOT,read,digest,require,atomic
from pipeline_finalize import load_dcp_model,state_audit
require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU_ONLY');require(not torch.cuda.is_initialized(),'NO_CUDA')
torch.set_num_threads(8);c=config();a2=read(c['stage2']['authority']);A=ROOT/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1'
parent=Path(a2['DEBUG_STAGE1_PARENT_PATH']);manifest=read(parent/'manifest.json');source=Path(manifest['metadata']['source_checkpoint'])
require(digest(source/'manifest.json')==manifest['metadata']['source_manifest_sha256'],'DEBUG_PARENT_SOURCE_IDENTITY')
state=load_dcp_model(source);audit1=state_audit(state,c)
reference=torch.load(parent/'model.pt',weights_only=True,map_location='cpu',mmap=True)
require(set(state)==set(reference),'PROVEN_MODEL_ONLY_KEYS')
for k in state:require(torch.equal(state[k],reference[k]),'PROVEN_MODEL_ONLY_ROUTE_MISMATCH '+k)
del state,reference;gc.collect()
from safetensors.torch import load_file
hf=Path(a2['DEBUG_STAGE2_HF_PATH']);state=load_file(str(hf/'model.safetensors'),device='cpu');audit2=state_audit(state,c)
sys.path.insert(0,str(ROOT/'src/inference_runtime_4k_stage2_v1/code'))
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
cfg=Qwen3Config.from_pretrained(hf,local_files_only=True);cfg._attn_implementation='sdpa'
with torch.device('meta'):model=Qwen3ForCausalLM(cfg)
expected=model.state_dict();require(set(expected)==set(state),'INFERENCE_META_KEYS')
for k,v in expected.items():require(v.shape==state[k].shape,'INFERENCE_META_SHAPE '+k)
require(sum(p.numel() for p in model.parameters())==4061881856,'INFERENCE_PARAMETER_COUNT')
require(cfg.ttt_chunk==4096 and cfg.ttt_lr==1.0 and cfg.ttt_update_clip_norm==1e-5,'INFERENCE_4K_PARAMETERS')
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(hf,local_files_only=True);require(len(tok.encode('CPU-only static compatibility fixture.'))>0,'STANDALONE_TOKENIZER')
require(not torch.cuda.is_initialized(),'CPU_ONLY_END')
result={'status':'PASS','created_at':time.time(),'DEBUG_ONLY':True,'formal_final_created':False,'GPU_used':False,'stage1_proven_DCP_to_model_only_exact':True,'stage1_keys':audit1['key_count'],'stage2_keys':audit2['key_count'],'unique_parameter_count':audit2['unique_parameter_count'],'stage1_state_dict_audit':audit1,'stage2_state_dict_audit':audit2,'inference_meta_architecture_compatible':True,'tokenizer_resolved':True,'inference_chunk':4096,'no_forward_or_RULER_executed':True,'debug_parent_manifest_sha256':digest(parent/'manifest.json'),'debug_inference_authority_sha256':digest(hf/'INFERENCE_PARENT_AUTHORITY.json'),'verification_source_sha256':digest(Path(__file__))}
atomic(A/'REAL_DEBUG_ARTIFACT_CPU_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if 'state_dict_audit' not in k},indent=2))
