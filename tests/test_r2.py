from itertools import product
from dynamic_ttt.mechanisms.runtime import r2_candidates,r2_winner
def test_bruteforce():
 actions=['OFF','L0','ALL'];n=4
 expected={x for x in product(actions,repeat=n) if sum(a!=b for a,b in zip(x,x[1:]))==1}
 actual={tuple(x['sequence']) for x in r2_candidates(n,[0])};assert actual==expected
 assert len(list(r2_candidates(4,[0,6,12,18,24,30])))==168
def test_ties(layers):
 rs=[dict(x,score=1.) for x in r2_candidates(4,layers)]
 w,score,ties=r2_winner(rs,layers);assert (w['tau'],w['A'],w['B'])==(1,'OFF','L0');assert ties==126
