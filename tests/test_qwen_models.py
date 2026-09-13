import torch,pytest
from dynamic_ttt.models.qwen3 import Qwen3Config,Qwen3ForCausalLM
from dynamic_ttt.models.actions import action_scope

@pytest.mark.parametrize('action,expected',[('OFF',[]),('L30',[30]),('ALL',[0,6,12,18,24,30])])
def test_real_qwen_action_cache_coverage(action,expected):
 torch.set_num_threads(2);torch.manual_seed(719)
 layers=[0,6,12,18,24,30]
 cfg=Qwen3Config(vocab_size=19,hidden_size=8,intermediate_size=12,num_hidden_layers=31,num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=2048,ttt_mode=True,ttt_layers=layers,ttt_chunk=1024,ttt_lr=1.,ttt_proj=True,ttt_target='hidden_states',ttt_target_type='gated_next_position_hidden',ttt_ntp_gate=True,ttt_update_clip_norm=1e-5)
 cfg._attn_implementation='sdpa';model=Qwen3ForCausalLM(cfg).eval()
 with torch.no_grad():
  for l in layers:model.model.layers[l].mlp.ttt_ntp_gate.fill_(.7)
  base=[model.model.layers[l].mlp.down_proj.weight.clone() for l in layers]
  with action_scope(model,action):result=model(torch.arange(1026).remainder(19).unsqueeze(0),use_cache=True)
  if expected:
   assert [l for l in layers if result.past_key_values.ttt_states[l][2] is not None]==expected
  for l,w in zip(layers,base):torch.testing.assert_close(w,model.model.layers[l].mlp.down_proj.weight,rtol=0,atol=0)
  assert all(hasattr(model.model.layers[l].mlp,'ttt_ntp_gate') for l in layers)
