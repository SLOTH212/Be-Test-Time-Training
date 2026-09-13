from dynamic_ttt.mechanisms.runtime import reset_screen
from dynamic_ttt.mechanisms.experimental.reset_resolution import screen
def test_historical_boundary(layers):
 p=reset_screen(['L0']+['L6']*15,layers)
 assert p['eligible'] and p['boundary']==2
 assert not reset_screen(['L0']*16,layers)['eligible']
def test_candidate_separate(layers):
 assert callable(screen)
