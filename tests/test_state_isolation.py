import torch
from model_helpers import tiny_model
from dynamic_ttt.models.llama31 import snapshot,restore

def test_snapshot_has_no_tensor_aliases():
    model=tiny_model()
    with torch.no_grad():cache=model(torch.tensor([[1,4,5,6,7]]),use_cache=True).past_key_values
    saved=snapshot(cache);before=saved[0][2].clone()
    cache.ttt_states[0][2].add_(1)
    torch.testing.assert_close(saved[0][2],before,rtol=0,atol=0)
    restore(cache,saved);cache.ttt_states[0][2].add_(1)
    torch.testing.assert_close(saved[0][2],before,rtol=0,atol=0)
