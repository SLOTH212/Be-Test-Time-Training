from dynamic_ttt.eval.runtime import sample_best,best_single,fixed_actions
import pytest
def test_all_and_off_included(layers):
 s={a:0. for a in fixed_actions(layers)};s['ALL']=1.
 assert sample_best(s,layers)==(1.,['ALL'])
 s['OFF']=1.;assert sample_best(s,layers)==(1.,['OFF','ALL'])
 assert best_single([s],layers)==('L0',0.)
def test_incomplete(layers):
 with pytest.raises(ValueError):sample_best({'OFF':0},layers)
