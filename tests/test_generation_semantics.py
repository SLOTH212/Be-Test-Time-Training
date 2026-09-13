import torch
from model_helpers import tiny_model

def test_generation_retains_prompt_fast_weights():
    model=tiny_model();ids=torch.tensor([[1,4,5,6,7,8,9,3,4,5]])
    with torch.no_grad():
        prompt=model(ids,use_cache=True)
        cache=prompt.past_key_values
        weight=cache.ttt_states[0][2].clone()
        output=model(torch.tensor([[7]]),past_key_values=cache,cache_position=torch.tensor([10]),use_cache=True)
        torch.testing.assert_close(output.past_key_values.ttt_states[0][2],weight,rtol=0,atol=0)
