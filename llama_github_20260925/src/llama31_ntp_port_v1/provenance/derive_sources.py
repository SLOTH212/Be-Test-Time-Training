"""Reproducible model-family port from the two frozen local references."""
from pathlib import Path
import ast, shutil
W=Path(__file__).resolve().parents[1]
Q=W/'reference/qwen_ntp'; R=W/'reference/inplace_ttt'
def node(text,name,parent=None):
 tree=ast.parse(text); nodes=tree.body if parent is None else next(n for n in tree.body if getattr(n,'name',None)==parent).body
 return next(n for n in nodes if getattr(n,'name',None)==name)
def get(text,name,parent=None):
 n=node(text,name,parent);return '\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])
def replace(text,name,new,parent=None):
 n=node(text,name,parent);lines=text.splitlines();return '\n'.join(lines[:n.lineno-1]+new.splitlines()+lines[n.end_lineno:])+'\n'
for d in ['hf_models','hf_models/hf_llama','inference_model','inference_model/hf_llama','ntp_core']:
 p=W/'source'/d;p.mkdir(parents=True,exist_ok=True);(p/'__init__.py').write_text('')
core=(Q/'hf_models/hf_qwen3/ttt_state_core.py').read_text()
# The sole common-core generalization: chunk size becomes a positive config integer.
core=core.replace('if chunk_size != 1024 and not allow_test_override:\n        raise ValueError("frozen Static-TTT chunk_size must be 1024")','if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:\n        raise ValueError("chunk_size must be a positive integer")')
(W/'source/ntp_core/ttt_state_core.py').write_text(core)
for mode,rd,qd,out in [('training','hf_models/hf_llama','hf_models/hf_qwen3','hf_models/hf_llama'),('inference','inference_model/hf_llama3','inference_model/hf_qwen3','inference_model/hf_llama')]:
 l=(R/rd/'modeling_llama.py').read_text();q=(Q/qd/'modeling_qwen3.py').read_text()
 mlp=get(q,'Qwen3MLP').replace('Qwen3','Llama')
 # Llama preserves its architecture's configurable MLP biases.
 for proj in ['gate_proj','up_proj','down_proj']:
  lines=mlp.splitlines();mlp='\n'.join(line.replace('bias=False','bias=config.mlp_bias') if 'self.'+proj+' = nn.Linear' in line else line for line in lines)
 l=replace(l,'LlamaMLP',mlp)
 l=replace(l,'_init_weights',get(q,'_init_weights','Qwen3PreTrainedModel'),'LlamaPreTrainedModel')
 # Qwen inference initializer does not explicitly initialize RMSNorm; Llama requires it.
 if mode=='inference':
  init=get(l,'_init_weights','LlamaPreTrainedModel')+'\n        elif isinstance(module, LlamaRMSNorm):\n            module.weight.data.fill_(1.0)'
  l=replace(l,'_init_weights',init,'LlamaPreTrainedModel')
 l=l.replace('from .configuration_llama import LlamaConfig','from .configuration_llama import LlamaConfig\nfrom ntp_core.ttt_state_core import GATED_NTP_TARGET, TTTState, run_document, run_sequence_with_boundaries\nfrom ntp_core.validation import validate_config')
 l=l.replace('self.padding_idx = config.pad_token_id','validate_config(config, "llama")\n        self.padding_idx = config.pad_token_id')
 if mode=='training':
  f=get(l,'forward','LlamaDecoderLayer').replace('target_states: Optional[torch.Tensor] = None,','target_states: Optional[torch.Tensor] = None,\n        ttt_valid_mask: Optional[torch.Tensor] = None,\n        document_boundaries=None,')
  f=f.replace('self.mlp(hidden_states, t=target_states)','self.mlp(hidden_states, t=target_states, valid_mask=ttt_valid_mask, document_boundaries=document_boundaries)')
  l=replace(l,'forward',f,'LlamaDecoderLayer')
  # Only require training masks on TTT-enabled paths, preserving vanilla cached OFF forward.
  start=q.index('        # Resolve the canonical token-valid mask');end=q.index('        # create position embeddings',start)
  block=q[start:end];block='        ttt_valid_mask = None\n        document_boundaries = kwargs.pop("document_boundaries", None)\n        if self.ttt_mode:\n'+ '\n'.join('    '+line for line in block.splitlines() if 'document_boundaries = kwargs.pop' not in line)+'\n'
  l=l.replace('        position_embeddings = self.rotary_emb(hidden_states, position_ids)',block+'        position_embeddings = self.rotary_emb(hidden_states, position_ids)')
  l=l.replace('target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),','target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),\n                ttt_valid_mask=ttt_valid_mask,\n                document_boundaries=document_boundaries,')
 else:
  cache=get(q,'TTTDynamicCache').replace('[(None, None, None)] * 100','[(None, None, None)] * config.num_hidden_layers')
  l=replace(l,'TTTDynamicCache',cache)
  forward=get(q,'forward','Qwen3DecoderLayer').replace('Qwen3','Llama')
  l=replace(l,'forward',forward,'LlamaDecoderLayer')
  l=l.replace('TTTDynamicCache()','TTTDynamicCache(config=self.config)')
  anchor='        for decoder_layer in self.layers[: self.config.num_hidden_layers]:'
  block='        ttt_generation_step = bool(\n            self.ttt_mode and cache_position is not None and cache_position.numel() == 1\n            and int(cache_position[0].item()) > 0\n        )\n\n'
  l=l.replace(anchor,block+anchor)
  l=l.replace('target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),','target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),\n                ttt_enabled=self.ttt_mode,\n                ttt_generation_step=ttt_generation_step,')
  l=replace(l,'generate',get(q,'generate','Qwen3ForCausalLM').replace('TTTDynamicCache()','TTTDynamicCache(config=self.config)'),'LlamaForCausalLM')
 # Biasful TTT requires a scientific decision; disabled vanilla bias configurations remain supported.
 dest=W/'source'/out
 (dest/'modeling_llama.py').write_text(l)
 cfg=(R/rd/'configuration_llama.py').read_text()
 cfg=cfg.replace('ttt_layers=[0, 6, 12, 18, 24, 30],','ttt_layers=None,').replace('ttt_mode=True,','ttt_mode=False,').replace('ttt_lr=0.3,','ttt_lr=1.0,').replace('ttt_chunk=8192,','ttt_chunk=1024,').replace('self.ttt_layers = ttt_layers','self.ttt_layers = [] if ttt_layers is None else list(ttt_layers)')
 (dest/'configuration_llama.py').write_text(cfg)
 (dest/'__init__.py').write_text('from .configuration_llama import LlamaConfig\nfrom .modeling_llama import LlamaForCausalLM, LlamaModel\n')
print('Derived separate Llama training and inference sources.')
