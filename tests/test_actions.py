from dynamic_ttt.eval.runtime import fixed_actions,action_layers
import pytest
def test_action_coverage(layers):
 assert fixed_actions(layers)==['OFF','L0','L6','L12','L18','L24','ALL']
 assert action_layers(layers+[30])['L30']==[30]
 assert action_layers(layers)['OFF']==[]
 assert action_layers(layers)['ALL']==layers
def test_invalid():
 with pytest.raises(ValueError):fixed_actions([6,0])
