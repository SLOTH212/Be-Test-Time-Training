"""Qwen3 model classes and per-sample fast-weight cache access."""
from dynamic_ttt.models.inference.hf_qwen3.configuration_qwen3 import Qwen3Config
from dynamic_ttt.models.inference.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, TTTDynamicCache
from dynamic_ttt.models.family import decoder_blocks
def down_projection(model, layer): return model.model.layers[layer].mlp.down_proj
def snapshot(cache):
 return [(None if h is None else h.detach().clone(), None if t is None else t.detach().clone(), None if w is None else w.detach().clone()) for h,t,w in cache.ttt_states]
def restore(cache, state):
 cache.ttt_states=[tuple(None if x is None else x.detach().clone() for x in row) for row in state]
