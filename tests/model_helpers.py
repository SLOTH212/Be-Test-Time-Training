import torch
from dynamic_ttt.models.llama31 import LlamaConfig, LlamaForCausalLM

def tiny_model():
    torch.manual_seed(719)
    config=LlamaConfig(vocab_size=37,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        ttt_mode=True,ttt_layers=[0],ttt_chunk=4,ttt_proj=True,ttt_lr=1.,
        ttt_target='hidden_states',ttt_target_type='gated_next_position_hidden',
        ttt_ntp_gate=True,ttt_update_clip_norm=1e-5)
    config._attn_implementation='sdpa'
    model=LlamaForCausalLM(config).eval()
    with torch.no_grad(): model.model.layers[0].mlp.ttt_ntp_gate.fill_(0.7)
    return model
