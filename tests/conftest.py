import pytest
@pytest.fixture
def layers():return [0,6,12,18,24]
@pytest.fixture
def tensors():
 import torch
 torch.manual_seed(719)
 return [torch.randn(*s,dtype=torch.float64) for s in [(1,4,5),(1,4,3),(3,5),(3,3),(3,)]]
