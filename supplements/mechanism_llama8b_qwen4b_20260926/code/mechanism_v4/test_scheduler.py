"""CPU-only integration checks; run on Linux next to run.py. No GPU/model access."""
from pathlib import Path
import copy,json,multiprocessing as mp,os,tempfile,time,unittest
import run as r

def record(can):
 return dict(can,score=.5,prediction_hash='a'*64,generation_update_count=0,kv_replaced_layers=[],
  observed_update_calls=sum(len(r.e.MAPPING[v]) for v in can['sequence']),
  layer_update_counts={str(l):sum(l in r.e.MAPPING[v] for v in can['sequence']) for l in r.e.LAYERS},seconds=.01)

class FakeEngine:
 base_hash='unchanged';model=None
 class torch:
  class cuda:
   @staticmethod
   def empty_cache():pass
 def __init__(self,c):pass
 def run(self,row,seq):
  can=next(v for v in r.e.candidates(len(seq)) if v['sequence']==seq)
  with (r.O/'attempts.jsonl').open('a') as f:f.write(json.dumps(can['full_schedule_hash'])+'\n');f.flush()
  time.sleep(.03)
  return record(can),{}

def run_worker(gpu):
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 r.worker('test',gpu)

class Scheduler(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  r.S=self.root/'source';r.S.mkdir();r.O=self.root/'output';r.O.mkdir()
  r.atomic(r.S/'JOB.json',{'test':True})
  self.x=dict(sample_id='task:0001',sequence=['L0']*6)
  self.t=dict(id='task_0001__r0',kind='r2',sample_id=self.x['sample_id'],start=0,end=16)
  self.job={'models':{'test':{'cohort':[self.x],'tasks':[self.t]}}}
  r.verify=lambda **kw:self.job;r.verify_import=lambda:None
  r.e.Engine=FakeEngine;r.e.param_hash=lambda _:FakeEngine.base_hash;r.e.getrow=lambda sid:{}
  self.ctx=mp.get_context('fork')
 def tearDown(self):self.tmp.cleanup()
 def start(self,gpu):
  p=self.ctx.Process(target=run_worker,args=(gpu,));p.start();return p
 def done(self,p):p.join(15);self.assertFalse(p.is_alive());self.assertEqual(p.exitcode,0)
 def test_competing_workers_execute_once(self):
  a=self.start(0);b=self.start(1);self.done(a);self.done(b)
  attempts=(r.O/'attempts.jsonl').read_text().splitlines()
  self.assertEqual(len(attempts),16);self.assertEqual(len(set(attempts)),16)
  self.assertEqual(len(r.read(r.receipt_path('test',self.t))['result']['records']),16)
 def test_kill_recovery_preserves_committed_prefix(self):
  p=self.start(0);checkpoint=r.O/'test'/'partial'/(self.t['id']+'.json')
  until=time.monotonic()+10
  while not checkpoint.exists() and time.monotonic()<until:time.sleep(.005)
  self.assertTrue(checkpoint.exists());p.terminate();p.join(10)
  prefix=r.read(checkpoint)['records'];self.assertGreaterEqual(len(prefix),8)
  saved=copy.deepcopy(prefix);self.done(self.start(1))
  result=r.read(r.receipt_path('test',self.t))['result']['records']
  self.assertEqual(result[:len(saved)],saved);self.assertEqual(len(result),16)
  self.assertEqual(len({v['full_schedule_hash'] for v in result}),16)
 def test_sparse_import_prefix_and_complete_skip(self):
  # Imported global prefix40: block32..64 must resume at40, not32 or64.
  self.t.update(id='task_0001__r32',start=32,end=64)
  cs=r.e.candidates(6);seed=r.O/'imported'/'test'/'task_0001'/'R2_PROGRESS.json'
  r.atomic(seed,{'records':[record(c) for c in cs[:40]],'completed':40,'total':len(cs)})
  self.done(self.start(0));out=r.read(r.receipt_path('test',self.t))['result']['records']
  self.assertEqual(out,[record(c) for c in cs[32:64]])
  self.assertEqual(len((r.O/'attempts.jsonl').read_text().splitlines()),24)
  self.done(self.start(1));self.assertEqual(len((r.O/'attempts.jsonl').read_text().splitlines()),24)
 def test_bad_checkpoint_rejected(self):
  path=r.O/'test'/'partial'/(self.t['id']+'.json')
  r.atomic(path,{'task':self.t,'job_sha256':'wrong','records':[]})
  p=self.start(0);p.join(10);self.assertNotEqual(p.exitcode,0)
  self.assertFalse(r.receipt_path('test',self.t).exists())
 def test_candidate_validation_and_tie_break(self):
  cs=r.e.candidates(6);rs=[record(c) for c in cs]
  r.validate_records(self.x,{'records':rs,'completed':len(rs),'total':len(cs)})
  self.assertEqual(r.best_summary(rs)['winner']['tau'],1)
  for mutate in [lambda q:q.reverse(),lambda q:q[0].update(generation_update_count=1),lambda q:q[0].update(score=float('nan'))]:
   bad=copy.deepcopy(rs);mutate(bad)
   with self.assertRaises(AssertionError):r.validate_records(self.x,{'records':bad,'completed':len(bad),'total':len(cs)})
  with self.assertRaises(AssertionError):r.validate_block(self.x,rs[:17],0,16)

if __name__=='__main__':unittest.main(verbosity=2)
