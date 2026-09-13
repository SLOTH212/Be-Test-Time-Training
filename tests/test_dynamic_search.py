from dynamic_ttt.dynamic import search
from dynamic_ttt.eval.runtime import replay_budget
def test_real_search_with_recomputed_fixture():
 class Replay:
  full_chunks=3
  def __init__(self):self.calls={}
  def evaluate(self,seq):
   seq=tuple(seq)
   if seq not in self.calls:
    state=0.
    for i,a in enumerate(seq):state=(state+search.ACTIONS.index(a))/(i+2)
    self.calls[seq]={'sequence':list(seq),'sequence_hash':search.canonical(list(seq)),'score':state}
   return self.calls[seq]
 ev=Replay();best,records=search.beam_search(ev,range(3));assert best['score']==max(r['score'] for r in records)
 assert len(ev.calls)<=replay_budget(7,3)
