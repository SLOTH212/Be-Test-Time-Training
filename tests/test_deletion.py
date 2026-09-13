from dynamic_ttt.mechanisms.runtime import deletion_plan
from dynamic_ttt.mechanisms.experimental.omit_update import deletion_target
import pytest
def test_distinct_variants(layers):
 p=deletion_plan(['L0']+['L6']*15,2,layers);assert p['selected_component_layers']==[0]
 q=deletion_target(['L0','L30'],1,30);assert q['omit_update_only']
 with pytest.raises(ValueError):deletion_target(['OFF'],0,0)
