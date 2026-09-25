import os,sys,json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
R=Path('/path/to/ttt');H=R/'work/qwen3_4b_stage2_preflight_v1/debug_stage2_hf';A=R/'audits/qwen3_4b_stage2_30m_32k_gpu23_preflight_v1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
torch.cuda.set_device(1);device=torch.device('cuda',1);assert torch.cuda.device_count()==2
sys.path.insert(0,str(R/'src/inference_runtime/code'))
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
cfg=Qwen3Config.from_pretrained(H,local_files_only=True)
assert cfg.ttt_layers==[0,6,12,18,24,30] and cfg.ttt_chunk==4096
model,info=Qwen3ForCausalLM.from_pretrained(H,config=cfg,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
model.to(device).eval();tok=AutoTokenizer.from_pretrained(H,local_files_only=True)
raw=tok('A short local checkpoint compatibility test.',return_tensors='pt',add_special_tokens=False).input_ids.to(device)
results=[]
for mode in ['OFF','L0']:
 stashed={};prior=model.model.ttt_mode
 for layer in cfg.ttt_layers:
  if mode=='L0' and layer==0:continue
  mlp=model.model.layers[layer].mlp;stashed[layer]={}
  for name in ['ttt_conv','ttt_proj','ttt_ntp_gate']:
   if hasattr(mlp,name):stashed[layer][name]=getattr(mlp,name);delattr(mlp,name)
 model.model.ttt_mode=mode!='OFF'
 ids=raw if mode=='OFF' else raw.repeat(1,(4097+raw.shape[1]-1)//raw.shape[1])[:,:4097]
 torch.cuda.reset_peak_memory_stats();before={n:p._version for n,p in model.named_parameters()};started=time.perf_counter()
 try:
  with torch.inference_mode():
   y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
  torch.cuda.synchronize();assert y.shape[1]==ids.shape[1]+1
  stats=getattr(model.model.layers[0].mlp,'last_ttt_stats',[]) if mode=='L0' else []
  if mode=='L0':assert len(stats)>0
  assert all(p._version==before[n] for n,p in model.named_parameters())
  results.append({'mode':mode,'status':'PASS','prompt_tokens':ids.shape[1],'generated_tokens':1,'ttt_stats_count':len(stats),'elapsed_seconds':time.perf_counter()-started,'peak_memory_bytes':torch.cuda.max_memory_reserved(),'physical_gpu':3,'parameter_versions_unchanged':True})
 finally:
  model.model.ttt_mode=prior
  for layer,attrs in stashed.items():
   for name,value in attrs.items():setattr(model.model.layers[layer].mlp,name,value)
receipt={'status':'PASS','DEBUG_ONLY':True,'PAPER_RESULT':False,'checkpoint':str(H),'loading_info':info,'results':results,'runtime':'ttt_inference_runtime_v1','runtime_package_sha256':'65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51','benchmark_run':False}
(A/'STAGE2_INFERENCE_HANDOFF_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
