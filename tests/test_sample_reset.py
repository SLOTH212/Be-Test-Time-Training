import torch
from model_helpers import tiny_model

def test_samples_are_independent():
    model=tiny_model();ids=torch.tensor([[1,4,5,6,7,8,9,3,4]])
    base={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        first=model(ids,use_cache=True).logits.clone()
        model(torch.tensor([[1,9,8,7,6,5,4,3,2]]),use_cache=True)
        last=model(ids,use_cache=True).logits
    torch.testing.assert_close(first,last,rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,base[k],rtol=0,atol=0)
