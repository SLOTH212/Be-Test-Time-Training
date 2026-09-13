def inference_classes(cfg):
 family=cfg.get('model_family','qwen3').lower()
 if 'llama' in family:
  from dynamic_ttt.models.inference.hf_llama.configuration_llama import LlamaConfig
  from dynamic_ttt.models.inference.hf_llama.modeling_llama import LlamaForCausalLM
  return LlamaConfig,LlamaForCausalLM
 if cfg['ttt_chunk_size']==4096:
  from dynamic_ttt.models.inference.hf_qwen3_scaleup.configuration_qwen3 import Qwen3Config
  from dynamic_ttt.models.inference.hf_qwen3_scaleup.modeling_qwen3 import Qwen3ForCausalLM
 else:
  from dynamic_ttt.models.inference.hf_qwen3.configuration_qwen3 import Qwen3Config
  from dynamic_ttt.models.inference.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
 return Qwen3Config,Qwen3ForCausalLM
