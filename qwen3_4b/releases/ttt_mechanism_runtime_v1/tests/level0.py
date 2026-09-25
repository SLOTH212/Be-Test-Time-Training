"""Stdlib-only structural, boundary, identity, and process-ownership tests."""
import ast,itertools,json,multiprocessing,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
from worker import Store
R=rt.ROOT;L=[0,6,12,18,24];S=L+[30]
def rejected(fn,exc=(ValueError,RuntimeError)):
 try:fn()
 except exc:return
 raise AssertionError('negative guard accepted invalid input')
def lock_attempt(path,q):
 s=Store(path,{'test':1})
 try:
  with s.own('sample'):q.put('ACQUIRED')
 except RuntimeError:q.put('BLOCKED')
def main():
 rt.dependency()
 for p in (R/'configs').glob('*.yaml'):rt.load_config(p)
 for layers in [L,S]:
  acts=rt.actions(layers)
  assert len(acts)==len(layers)+2 and rt.mapping(layers)['ALL']==layers
  for n in [8,12,15,16,17]:
   for first in range(1,n):
    seq=[acts[1]]*first+[acts[2]]*(n-first);p=rt.reset_screen(seq,layers)
    assert p['eligible']==((first+1)/n<=.25 and max(seq[n//2:].count(a) for a in acts)/len(seq[n//2:])>=.8)
    if p['eligible']:assert p['boundary']==first+1
   seq=[acts[1]]+[acts[2]]*(n-1);assert rt.reverse(rt.reverse(seq))==seq
  # T=10 second-half denominator=5; exactly 4/5 passes, 3/5 fails.
  assert rt.reset_screen(['L0','L6','L6','L6','L6','L6','L6','L6','L6','OFF'],layers)['eligible']
  assert not rt.reset_screen(['L0','L6','L6','L6','L6','L6','L6','L6','OFF','OFF'],layers)['eligible']
  tie=rt.reset_screen(['L0','L6','L6','L6','OFF','L6','OFF','L6'],layers)
  assert tie['details']['steady_tail_action']=='OFF' and not tie['eligible']
  assert rt.reset_screen(['L0']*8,layers)['exclusion_reason']=='NO_ACTION_SWITCH'
  seq=['ALL']+['L0']*7;p=rt.deletion_plan(seq,2,layers);assert p['selected_component_layers']==layers and p['sham_layer'] is None
  p=rt.deletion_plan(['L6']+['L0']*7,2,layers);assert p['selected_component_layers']==[6] and p['sham_layer']==0
  assert not rt.deletion_plan(['OFF']+['L0']*7,2,layers)['eligible']
  kp=rt.kp_plan([acts[-2]]+['L0']*7,2,layers);assert kp['cells']['F00']['injection_layers']==layers;assert kp['cells']['F11']['injection_layers']==[0,layers[-1]]
  candidates=list(rt.r2_candidates(8,layers));assert len(candidates)==len(acts)*(len(acts)-1)*7
  assert len({x['full_schedule_hash'] for x in candidates})==len(candidates)
  for x in candidates:assert sum(a!=b for a,b in zip(x['sequence'],x['sequence'][1:]))==1
  records=[{**x,'score':1.0} for x in reversed(candidates)];w,b,t=rt.r2_winner(records,layers);assert (w['tau'],w['A'],w['B'])==(1,'OFF','L0') and t==len(candidates)
 assert len(list(rt.r2_candidates(8,S)))==392
 # Historical cohort membership is rebuilt from scalar inputs, never hand-selected.
 raw=rt.rows(R/'tests/reference/dynamic_fixed_scalar_inputs.jsonl');fixed={x['sample_id']:x['fixed_scores'] for x in raw};dynamic={x['sample_id']:x for x in raw};parent=rt.improved_cohort(fixed,dynamic,L);expected=rt.rows(R/'provenance/IMPROVED_173_PARENT_COHORT.jsonl')
 assert {x['sample_id'] for x in parent}=={x['sample_id'] for x in expected} and len(parent)==173
 reset=[x for x in parent if rt.reset_screen(x['action_sequence'],L)['eligible']];assert len(reset)==167
 deletion=[x for x in reset if rt.deletion_plan(x['action_sequence'],rt.reset_screen(x['action_sequence'],L)['boundary'],L)['eligible']];assert len(deletion)==151
 rejected(lambda:rt.validate_result({}))
 cfg=rt.load_config(R/'configs/qwen3_1p7b_reverse_formal_reference.yaml');bad={**cfg,'actions':cfg['actions']+['L30']};rejected(lambda:rt.validate_config(bad));rejected(lambda:rt.actions([0,0]));rejected(lambda:rt.r2_winner([],L),ValueError)
 with tempfile.TemporaryDirectory(prefix='mechanism_test_',dir=os.environ.get('TTT_TEST_TMP_ROOT',R/'audits')) as tmp:
  s=Store(tmp,{'test':1});rejected(lambda:Store(tmp,{'test':2}))
  with s.own('sample'):
   ctx=multiprocessing.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=lock_attempt,args=(tmp,q));p.start();p.join(15);assert p.exitcode==0 and q.get(timeout=1)=='BLOCKED'
   s.commit('sample',{'value':1});rejected(lambda:s.commit('sample',{'value':2}))
  s2=Store(tmp,{'test':1});assert s2.committed('sample')['value']==1
  with s2.own('sample'):pass
  data=json.loads(s.path('sample').read_text());data['value']=2;s.path('sample').write_text(json.dumps(data));rejected(lambda:s.committed('sample'))
  false=Path(tmp)/'badarchive';false.write_bytes(b'not frozen archive');rejected(lambda:rt.dependency(archive=false))
 for p in R.glob('*.py'):ast.parse(p.read_text())
 print(json.dumps({'MECHANISM_RUNTIME_LEVEL0':'PASS','historical_cohorts':[173,167,151],'r2_32k_candidates':392,'sample_ownership_process_test':'PASS','config_resume_guard':'PASS'}))
if __name__=='__main__':main()
