"""Llama Dynamic: source-extracted frozen search with explicit model/4K bindings."""
import sys,os,json,hashlib,time,types,ast,collections,statistics,subprocess,fcntl,argparse,traceback,datetime,itertools
from pathlib import Path
R=Path('/path/to/ttt');S=R/'src/llama31_dynamic_v2_no_cwe_v1';C=R/'configs/llama31_dynamic_v2_no_cwe_v1.json'
sys.path.insert(0,str(R/'src/llama31_fixed8_formal_v1'));import runner as fixed
ACTIONS=fixed.ACTIONS;LAYERS=fixed.LAYERS
def config(full=False):
 c=fixed.read(C);assert fixed.sha(C)==C.with_suffix('.sha256').read_text().strip()
 if c.get('executor_version')=='V2' and c.get('fixed_baseline_status')!='READY':raise RuntimeError('V2_FIXED_BASELINE_REQUIRED: fresh Fixed8 results and cohort binding must precede formal Dynamic')
 for p,h in c['bindings'].items():assert fixed.sha(p)==h,('BINDING',p)
 assert c['excluded_tasks']==['ruler_cwe_16k'] and c['candidate_count']==570 and c['cwe_candidates']==0
 assert fixed.sha(c['fixed_authority'])==c['fixed_authority_sha256']
 fc=fixed.verify(weights=full);assert fixed.sha(fixed.C)==c['fixed_config_sha256']
 a=fixed.read(c['fixed_authority']);assert a['status']=='PASS' and a['sample_n']==6500 and a['cells']==52000
 if full:
  for p,h in a['commits'].items():assert fixed.sha(p)==h,('FIXED_RECEIPT_HASH',p)
 return c,fc
def source_runtime(c):
 path=Path(c['search_reference']);tree=ast.parse(path.read_text());names={'now','canonical','tensor_hash','install_dynamic_forward','Evaluator','hamming','diverse_top','constants','beam_search','baseline_fixed','verify_constants','morphology','search_sample'}
 class Adapt(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='hf_models.hf_qwen3.ttt_state_core':n.module='ntp_core.ttt_state_core'
   return n
  def visit_Constant(self,n):
   if type(n.value)==int and n.value==1024:return ast.copy_location(ast.Constant(4096),n)
   return n
 nodes=[Adapt().visit(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 ns=dict(json=json,hashlib=hashlib,time=time,types=types,Path=Path,defaultdict=collections.defaultdict,datetime=datetime.datetime,TOL=1e-12,ACTIONS=ACTIONS,LAYERS=LAYERS,ACTION_LAYERS={a:LAYERS if a=='ALL' else [] if a=='OFF' else [int(a[1:])] for a in ACTIONS},CHECKPOINT_HASH=c['model_sha256'],atomic_json=fixed.atomic,base_hash=fixed.param_hash)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
 sys.path.insert(0,str(R/'src/reference_executor_v2_port'))
 from adapter import bind_namespace
 bind_namespace(ns,'ntp_core.ttt_state_core',fixed.read(fixed.C)['chunk_size'])
 return types.SimpleNamespace(**{k:ns[k] for k in names})
def inputs(c,fc):
 rows=fixed.load_rows(fc);baseline={a:{} for a in ACTIONS};authority=fixed.read(c['fixed_authority']);by={r['sample_id']:r for r in rows}
 for p in authority['commits']:
  x=fixed.read(p);baseline[x['action']][x['sample_id']]={**x,'official_sample_score':x['score']}
 ids=fixed.read(c['cohort']);assert len(ids)==570 and all(by[s]['task'] not in c['excluded_tasks'] for s in ids)
 expected={r['sample_id'] for r in rows if r['task'] not in c['excluded_tasks'] and max(baseline[a][r['sample_id']]['score'] for a in ACTIONS)<1-1e-12};assert set(ids)==expected
 assignment=fixed.read(c['assignments']);flat=[s for ss in assignment.values() for s in ss];assert len(flat)==len(set(flat))==570 and set(flat)==set(ids)
 for row in rows:row.update(input=row['prompt'],_sample_index=row['global_index'],sample_index=int(row['sample_id'].split(':')[-1]))
 return by,baseline,ids
def cpu_tests(c):
 rt=source_runtime(c)
 class Ev:
  full_chunks=2
  def __init__(self):self.cache={}
  def evaluate(self,seq):
   seq=tuple(seq)
   if seq not in self.cache:self.cache[seq]={'sequence':list(seq),'sequence_hash':rt.canonical(list(seq)),'score':float(seq==('L30','ALL'))}
   return self.cache[seq]
 ev=Ev();best,recs=rt.beam_search(ev,range(2));assert best['score']==1 and best['sequence']==['L30','ALL'];assert len(recs)==64 and len(ev.cache)==64
 # Check exact source-selected sequence order for a tied objective and memoization.
 ev2=Ev();best2,recs2=rt.beam_search(ev2,range(2));assert [x['sequence'] for x in recs2]==[x['sequence'] for x in recs]
 assert len(rt.constants(7))==8 and rt.constants(7)[6]==('L30',)*7
 fixed.atomic(S/'CPU_TESTS.json',dict(status='PASS',full_two_chunk_enumeration=64,L30_and_ALL_covered=True,deterministic=True,config_sha256=fixed.sha(C)))
def worker(args):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and args.rank in range(4)
 c,fc=config();root=Path(c['smoke_root'] if args.smoke else c['output_root']);by,baseline,ids=inputs(c,fc)
 score_ns={};sp=Path(fc['scorer']);exec(compile(sp.read_text(),str(sp),'exec'),score_ns)
 model,tok,core=fixed.model_load(fc,args.rank);before=fixed.param_hash(model);versions={n:p._version for n,p in model.named_parameters()};rt=source_runtime(c);rt.install_dynamic_forward(model)
 ctx=types.SimpleNamespace(score_fn=score_ns['per_sample_score'],baseline=baseline);identity=fixed.canonical({'config':fixed.sha(C),'smoke':args.smoke})
 fixed.atomic(root/f'worker{args.rank}_loaded.json',dict(pid=os.getpid(),rank=args.rank,strict_load='PASS',model_hash=before,config_sha256=fixed.sha(C)))
 if args.smoke:
  tasks=['ruler_qa_squad_16k','ruler_fwe_16k','ruler_qa_hotpotqa_16k','ruler_vt_16k'];sid=next(s for s in ids if by[s]['task']==tasks[args.rank]);row=by[sid]
  ev=rt.Evaluator(ctx,model,tok,row,'SMOKE_CONSTANTS',1,1,root/f'worker{args.rank}_progress.json');recs,errors=rt.verify_constants(ctx,ev);assert not errors,errors
  seq=tuple(['L0','OFF','L30','ALL','L24','L6','L12'][:ev.full_chunks]);other=tuple(['ALL']*ev.full_chunks)
  a=ev.evaluate(seq,capture=True,force=True);ev.evaluate(other,force=True);b=ev.evaluate(seq,capture=True,force=True)
  assert a['prediction_hash']==b['prediction_hash'] and a['score']==b['score']
  assert [x['post_state_hash'] for x in a['chunk_layer_events']]==[x['post_state_hash'] for x in b['chunk_layer_events']];ev.finish()
  # Bounded end-to-end GPU search on a one-chunk synthetic fixture.
  tiny=dict(row);tiny['sample_id']='SMOKE_ONE_CHUNK_'+str(args.rank);tiny['_sample_index']=-1;tiny['input']='A quiet laboratory records neutral observations. '*180;tiny['max_new_tokens']=2
  # Extend a short fixture to exactly one full chunk plus a tail, using token IDs directly.
  ids0=tok(tiny['input'],add_special_tokens=True)['input_ids'];probe_ids=(ids0+ids0[1:]*10)[:4113]
  import torch
  class Tokens:
   eos_token_id=tok.eos_token_id
   def __call__(self,*a,**k):return types.SimpleNamespace(input_ids=torch.tensor([probe_ids]),items=lambda:[('input_ids',torch.tensor([probe_ids]))])
   def decode(self,*a,**k):return tok.decode(*a,**k)
  tinyctx=types.SimpleNamespace(score_fn=ctx.score_fn,baseline={a:{} for a in ACTIONS});tev=rt.Evaluator(tinyctx,model,Tokens(),tiny,'SMOKE_SEARCH',1,1,root/f'worker{args.rank}_toy_progress.json')
  assert tev.full_chunks==1
  # Evaluator progress needs baseline even during generation of reference constants.
  for act in ACTIONS:tinyctx.baseline[act][tiny['sample_id']]={'official_sample_score':0,'prediction':''}
  for act,seq0 in zip(ACTIONS,rt.constants(1)):
   x=tev.evaluate(seq0,capture=True);tinyctx.baseline[act][tiny['sample_id']]={'official_sample_score':x['score'],'prediction':x['prediction']}
  result=rt.search_sample(tinyctx,model,Tokens(),tiny,'SMOKE_SEARCH',1,1,root/f'worker{args.rank}_toy_progress.json');assert result['candidate_count']==8
  assert fixed.param_hash(model)==before and versions=={n:p._version for n,p in model.named_parameters()}
  fixed.atomic(root/f'worker{args.rank}_done.json',dict(status='PASS',constant_replays=8,mixed_trajectory_replays=3,exact_fixed_score_prediction=True,cross_branch_state_hashes_equal=True,one_chunk_search_candidates=8,base_unchanged=True,max_allocated_bytes=torch.cuda.max_memory_allocated(),config_sha256=fixed.sha(C)));return
 owned=fixed.read(c['assignments'])[str(args.rank)]
 for ordinal,sid in enumerate(owned,1):
  path=root/'samples'/(sid.replace(':','_')+'.json')
  if fixed.receipt(path,identity):continue
  start=time.time();result=rt.search_sample(ctx,model,tok,by[sid],'FORMAL_DYNAMIC_LB',ordinal,len(owned),root/f'worker{args.rank}_progress.json')
  n=result['full_chunk_count'];assert result['candidate_count']<=8+56+(n-1)*28 and result['dynamic_search_score']+1e-12>=result['fixed_oracle_score'];assert versions=={n:p._version for n,p in model.named_parameters()}
  result.update(DEBUG_ONLY=False,completed=True,FORMAL_RESULTS=True,wall_seconds=time.time()-start,parent_stage2_sha256=c['parent_stage2_sha256'],fixed_authority_sha256=c['fixed_authority_sha256'],chunk_size=4096,actions=ACTIONS,ttt_layers=LAYERS)
  fixed.commit(path,result,identity);fixed.atomic(root/f'worker{args.rank}_commits.json',dict(completed=ordinal,total=len(owned),sample_id=sid,fixed=result['fixed_oracle_score'],dynamic=result['dynamic_search_score'],seconds=time.time()-start,time=time.time()));print(json.dumps({'sample':sid,'fixed':result['fixed_oracle_score'],'dynamic':result['dynamic_search_score'],'candidates':result['candidate_count']}),flush=True)
 assert fixed.param_hash(model)==before;fixed.atomic(root/f'worker{args.rank}_done.json',dict(status='PASS',base_unchanged=True,samples=len(owned),time=time.time()))
def finalize(c,root):
 identity=fixed.canonical({'config':fixed.sha(C),'smoke':False});files=list((root/'samples').glob('*.json'));rows=[fixed.receipt(p,identity) for p in files];assert {r['sample_id'] for r in rows}==set(fixed.read(c['cohort'])) and len(rows)==570
 ag=fixed.read(Path(c['fixed_authority']).parent/'AGGREGATE.json');searched={r['sample_id']:r for r in rows};allrows=[]
 for f in ag['per_sample']:
  if f['task'] in c['excluded_tasks']:continue
  d=searched.get(f['sample_id']);score=d['dynamic_search_score'] if d else f['sample_best'];assert d or abs(f['sample_best']-1)<=1e-12
  allrows.append(dict(sample_id=f['sample_id'],task=f['task'],off=f['OFF'],sample_best=f['sample_best'],dynamic=score,searched=d is not None))
 aggregate={'samples':6000,'searched':570,'skipped_ceiling':5430,'included_cwe':False,'means':{k:statistics.mean(x[k] for x in allrows) for k in ['off','sample_best','dynamic']},'per_task':{},'per_sample':allrows}
 for t in sorted({r['task'] for r in allrows}):aggregate['per_task'][t]={k:statistics.mean(x[k] for x in allrows if x['task']==t) for k in ['off','sample_best','dynamic']}
 fixed.atomic(root/'AGGREGATE.json',aggregate);fixed.atomic(root/'DYNAMIC_FINAL_AUTHORITY.json',dict(schema='LLAMA31_DYNAMIC_FINAL_AUTHORITY_V1',status='PASS',DEBUG_ONLY=False,config_sha256=fixed.sha(C),fixed_authority_sha256=c['fixed_authority_sha256'],included_cwe=False,searched=570,population=6000,aggregate_sha256=fixed.sha(root/'AGGREGATE.json'),commits={str(p):fixed.sha(p) for p in files}))
def supervise(args):
 c,fc=config(full=True);root=Path(c['smoke_root'] if args.smoke else c['output_root']);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'manager.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (root/'PHASE_STATUS.json').exists()
 if not args.smoke:
  q=fixed.read(Path(c['smoke_root'])/'PHASE_STATUS.json');assert q['status']=='COMPLETE' and q['config_sha256']==fixed.sha(C)
 locks=[]
 for gpu in range(4):
  f=(R/'work/downstream_gpu_ownership'/f'physical_gpu{gpu}.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 import csv
 text=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 for line in csv.reader(text.splitlines()):assert line[1].strip().removeprefix('GPU-')==fixed.UUIDS[int(line[0])] and int(line[2])>=43000
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');ps=[];logs=[]
 for rank in range(4):
  f=(root/f'worker{rank}.log').open('a');logs.append(f);ps.append(subprocess.Popen([sys.executable,'-B',str(S/'dynamic.py'),'worker','--rank',str(rank)]+(['--smoke'] if args.smoke else []),env=env,stdout=f,stderr=subprocess.STDOUT))
 fixed.atomic(root/'PHASE_STATUS.json',dict(status='RUNNING',manager_pid=os.getpid(),pids=[p.pid for p in ps],started=time.time(),config_sha256=fixed.sha(C),smoke=args.smoke))
 while any(p.poll() is None for p in ps):
  if any(p.poll() not in [None,0] for p in ps):
   for p in ps:
    if p.poll() is None:p.terminate()
   break
  time.sleep(5)
 codes=[p.wait() for p in ps]
 if codes==[0]*4 and not args.smoke:finalize(c,root)
 fixed.atomic(root/'PHASE_STATUS.json',dict(status='COMPLETE' if codes==[0]*4 else 'FAILED',exit_codes=codes,ended=time.time(),config_sha256=fixed.sha(C),smoke=args.smoke));assert codes==[0]*4
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','worker','supervise']);p.add_argument('--rank',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.mode=='cpu':cpu_tests(config()[0])
 elif a.mode=='worker':worker(a)
 else:supervise(a)
if __name__=='__main__':main()
