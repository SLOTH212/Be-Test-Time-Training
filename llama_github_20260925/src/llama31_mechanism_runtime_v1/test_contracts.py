import unittest,itertools,tempfile,json,copy
from pathlib import Path
import mechanism as m
class Contracts(unittest.TestCase):
 def row(self):return m.fixture('test',m.FIXTURES['G'],[1]*16384)
 def test_01_order(self):self.assertEqual(m.rt.actions(m.LAYERS),m.ACTIONS)
 def test_02_l30(self):self.assertEqual(m.action_layers('L30'),[30])
 def test_03_all(self):self.assertEqual(m.action_layers('ALL'),[0,6,12,18,24,30])
 def test_04_off(self):self.assertEqual(m.action_layers('OFF'),[])
 def test_05_unknown(self):
  for a in ['L5','L24_ONLY','Fixed7','off']:
   with self.assertRaises(ValueError):m.action_layers(a)
 def test_06_schema(self):self.assertEqual(m.validate_trajectory(self.row())['complete_decision_chunks'],4)
 def test_07_fixed7(self):
  r=self.row();r['actions'].remove('L30')
  with self.assertRaises(ValueError):m.validate_trajectory(r)
 def test_08_final_layer(self):
  r=self.row();r['ttt_layers']=m.LAYERS[:-1]
  with self.assertRaises(ValueError):m.validate_trajectory(r)
 def test_09_model(self):
  r=self.row();r['model_revision']='qwen'
  with self.assertRaises(ValueError):m.validate_trajectory(r)
 def test_10_counts(self):
  for count in [0,3,5,True]:
   r=self.row();r['complete_decision_chunks']=count
   with self.assertRaises(ValueError):m.validate_trajectory(r)
 def test_11_tail(self):
  r=self.row();r['prompt_token_count']+=17;self.assertEqual(m.validate_trajectory(r)['complete_decision_chunks'],4)
 def test_12_reverse(self):
  for seq in m.FIXTURES.values():self.assertEqual(m.rt.reverse(m.rt.reverse(seq)),seq);self.assertEqual(sorted(seq),sorted(m.rt.reverse(seq)))
 def test_13_reset6(self):self.assertTrue(m.reset_plan(m.FIXTURES['D'])['eligible'])
 def test_14_reset7(self):self.assertTrue(m.reset_plan(m.FIXTURES['E'])['eligible'])
 def test_15_reset8(self):self.assertTrue(m.reset_plan(['L30']+['OFF']*7)['eligible'])
 def test_16_reset_constant(self):self.assertFalse(m.reset_plan(['L12']*8)['eligible'])
 def test_17_reset_late(self):self.assertFalse(m.reset_plan(['L30']*3+['OFF']*5)['eligible'])
 def test_18_reset_compatibility(self):
  old=m.rt.reset_rules(m.LAYERS).historical_early_steady
  for n in range(8,65):
   for switch in range(1,n):
    for a,b in [('OFF','L30'),('ALL','L6')]:
     seq=[a]*switch+[b]*(n-switch);self.assertEqual(old(seq)[0],m.reset_plan(seq)['eligible'])
 def test_19_reset_modal_tie(self):self.assertEqual(m.v2.screen(['L30','OFF','L30','OFF'],m.ACTIONS)['late_modal'],'OFF')
 def test_20_kp(self):
  p=m.kp_plan(['L30']+['L6']*6)
  self.assertEqual(p['control_sequence'],['OFF']+['L6']*6)
  for name,k,pbit in [('F00',0,0),('F10',1,0),('F01',0,1),('F11',1,1)]:
   c=p['cells'][name];self.assertEqual((c['K_bit'],c['P_bit']),(k,pbit));self.assertEqual(c['injection_layers'],[6,30] if pbit else m.LAYERS)
 def test_21_deletion(self):self.assertEqual(m.deletion_target(['L0','L30','L12','L18'],1,30)['chunk_index_0'],1)
 def test_22_deletion_invalid(self):
  for j,l in [(-1,30),(4,30),(1,24)]:
   with self.assertRaises(ValueError):m.deletion_target(['L0','L30','L12','L18'],j,l)
 def test_23_r2_exact(self):
  expected={s for s in itertools.product(m.ACTIONS,repeat=4) if sum(s[j]!=s[j-1] for j in range(1,4))==1}
  actual=m.r2_candidates(4);self.assertEqual(len(actual),168);self.assertEqual(len(actual),len({tuple(c['sequence']) for c in actual}));self.assertEqual(expected,{tuple(c['sequence']) for c in actual})
 def test_24_r2_tie(self):
  records=[{**c,'score':1.} for c in reversed(m.r2_candidates(4))];winner,best,n=m.rt.r2_winner(records,m.LAYERS);self.assertEqual((winner['tau'],winner['A'],winner['B'],n),(1,'OFF','L0',168))
 def test_25_r2_l30(self):self.assertTrue(any('L30' in c['sequence'] for c in m.r2_candidates(4)))
 def test_26_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=m.Receipts(tmp,'a');Path(tmp,'.x.partial.tmp').write_text('interrupted');self.assertIsNone(r.completed('x'));one=r.commit('x',{'v':1});self.assertEqual(r.commit('x',{'v':1}),one);self.assertEqual(len(list(Path(tmp).glob('*.json'))),1)
   with self.assertRaises(ValueError):r.commit('x',{'v':2})
 def test_27_resume_authority(self):
  with tempfile.TemporaryDirectory() as tmp:
   m.Receipts(tmp,'a').commit('x',{})
   with self.assertRaises(ValueError):m.Receipts(tmp,'b').completed('x')
 def test_28_worker_ownership(self):
  ids=[f'sample_{i}' for i in range(100)];self.assertEqual({m.owner(i) for i in ids},{0,1,2,3});self.assertEqual([m.owner(i) for i in ids],[m.owner(i) for i in ids])
 def test_29_formal_fail_closed(self):
  with self.assertRaises(ValueError):m.handoff(None,None,False)
 def test_30_fixture_coverage(self):self.assertEqual(set(m.FIXTURES),'ABCDEFG'.__iter__() if False else set('ABCDEFG'))
 def test_31_handoff(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'weight.txt').write_text('CPU TEST ONLY');stage={'schema':'LLAMA_STAGE2_FINAL_AUTHORITY_V1','status':'FINAL','model_revision':m.REVISION,'DEBUG_ONLY':False,'model_path':tmp,'files':{'weight.txt':m.sha(p/'weight.txt')}};m.atomic(p/'stage.json',stage)
   row=self.row();row.update(DEBUG_ONLY=False,completed=True,parent_stage2_sha256=m.sha(p/'stage.json'));(p/'rows.jsonl').write_text(json.dumps(row)+'\n')
   a={'schema':'LLAMA_DYNAMIC_TRAJECTORY_AUTHORITY_V1','DEBUG_ONLY':False,'FORMAL_RESULTS':True,'parent_stage2_sha256':m.sha(p/'stage.json'),'chain':['Stage2_FINAL','Fixed8','SampleBest','Dynamic_LB','improved_trajectories'],'trajectory_path':str(p/'rows.jsonl'),'trajectory_sha256':m.sha(p/'rows.jsonl')};m.atomic(p/'authority.json',a)
   self.assertEqual(len(m.handoff(p/'stage.json',p/'authority.json',True)[2]),1)
   a['DEBUG_ONLY']=True;m.atomic(p/'authority.json',a)
   with self.assertRaises(ValueError):m.handoff(p/'stage.json',p/'authority.json',True)
if __name__=='__main__':unittest.main(verbosity=2)
