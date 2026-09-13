import torch
from model_helpers import tiny_model

def test_partial_tail_uses_latest_state_without_update():
    model=tiny_model();mlp=model.model.layers[0].mlp
    x=torch.randn(1,10,16)
    with torch.no_grad():
        output,weight=mlp(x,x)
        h=mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
        torch.testing.assert_close(output[:,8:],torch.nn.functional.linear(h[:,8:],weight))
        assert sum(s.update_count for s in mlp.last_ttt_stats)==2
