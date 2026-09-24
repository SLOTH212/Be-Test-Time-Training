import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,csv,hashlib,collections,shutil,datetime,itertools
import numpy as np
from sampler import sample,render,decode,encode,seed,ACTIONS
R=Path(__file__).resolve().parent;B=Path('/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918');M=Path('/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921');roots=[Path('/home/zonghan/ttt/runs/formal/dynamic_old1k_v2_noncwe_local_20260918'),M/'dynamic_old1k_v2_remote_transfer_20260921/dynamic_old1k_v2_remote_results']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def put(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def tsv(n,rows):
 with (R/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
assert not (R/'RANDOM_PLAN.jsonl').exists(),'PLAN_ALREADY_FROZEN'
p=json.loads((B/'PROTOCOL.json').read_text());ph=sha(B/'PROTOCOL.json');merge=json.loads((M/'MERGED_12TASK.json').read_text());assert merge['status']=='COMPLETE' and merge['n_searched']==1433 and merge['protocol_hash']==ph
srcs=json.loads((M/'RESULT_SOURCES.json').read_text());allresults={};authority={}
for root,label in zip(roots,['local','remote']):
 assert json.loads((root/'COMPLETE.json').read_text())['status']=='COMPLETE'
 assert not list((root/'failures').glob('*')) and not (root/'supervisor_failure.json').exists()
 expected=set(json.loads((B/f'data/assignment_{label}.json').read_text()));got=set()
 for f in root.rglob('*'):
  if f.is_file():authority[str(f)]=sha(f)
 for f in (root/'results').glob('*.json'):
  x=json.loads(f.read_text());sid=x['sample_id'];assert sid not in allresults and authority[str(f)]==srcs[sid]['sha256'];assert x['protocol_hash']==ph and x['model_sha256']==p['model_sha256'] and x['executor_sha256']==p['executor_sha256'];got.add(sid);allresults[sid]=x
 assert got==expected
for f in [M/'FINAL_AUDIT.json',M/'MERGED_12TASK.json',M/'RESULT_SOURCES.json',M/'FINAL_PER_SAMPLE_6000.csv',B/'pipeline.py',B/'PROTOCOL.json',B/'MANIFEST.json',*list((B/'runtime').rglob('*.py')),*list((B/'data').glob('*'))]:
 if f.is_file():authority[str(f)]=sha(f)
assert sha(B/'runtime/executor.py')==p['executor_sha256'];assert sha(B/'model/model.safetensors')==p['model_sha256']
ct=collections.Counter('gt' if x['dynamic_score']>x['sample_best'] else 'eq' if x['dynamic_score']==x['sample_best'] else 'lt' for x in allresults.values());assert len(allresults)==1433 and ct['lt']==0
co=[];bud=[];inputs=[]
for sid,x in sorted(allresults.items()):
 if x['dynamic_score']<=x['sample_best']:continue
 cs=x['candidates'];seqs=[tuple(z['sequence']) for z in cs];T=len(x['winner']['sequence']);assert len(cs)==x['candidate_count']==len(set(seqs));const={s for s in seqs if len(set(s))==1};assert const=={tuple([a]*T) for a in ACTIONS}
 for z in cs:
  assert len(z['sequence'])==T and set(z['sequence'])<=set(ACTIONS) and z['clean_full_replay'] and z['recompute_downstream_updates'] and not z['stored_delta_stitching'] and z['generation_update_count']==0 and z['latency_seconds']>0
  assert z['checkpoint_hash']==p['model_sha256'] and z['ttt_chunk_size']==1024 and 0<=z['score']<=1
  assert z['sequence_hash']==hashlib.sha256(json.dumps(z['sequence'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 n=len(cs)-7;assert n>0
 co.append(dict(sample_id=sid,task=x['task'],num_decision_chunks=T,sample_best_score=x['sample_best'],dynamic_score=x['dynamic_score'],dynamic_gain=x['dynamic_score']-x['sample_best'],dynamic_winner_trajectory=json.dumps(x['winner']['sequence']),dynamic_prediction_hash=x['winner']['prediction_hash']))
 bud.append(dict(sample_id=sid,task=x['task'],T=T,total_candidate_records=len(cs),constant_unique_n=7,nonconstant_unique_evaluated_n=n,duplicate_n=0,failed_n=0,retry_n=0,B_i=n))
 inputs.append(dict(sample_id=sid,T=T,B_i=n))
tsv('RANDOM_BASELINE_COHORT.tsv',co);tsv('DYNAMIC_BUDGET_AUDIT.tsv',bud);put('SAMPLER_INPUT.json',inputs);put('AUTHORITY_HASHES.json',authority)
put('PREPLAN_FREEZE.json',dict(at=datetime.datetime.now().astimezone().isoformat(),cohort_sha256=sha(R/'RANDOM_BASELINE_COHORT.tsv'),budget_sha256=sha(R/'DYNAMIC_BUDGET_AUDIT.tsv'),sampler_inputs_sha256=sha(R/'SAMPLER_INPUT.json'),master_seed=20260923,replicate=0,strict_equality=True,gpu_execution_started=False))
# CPU P1: bijection, exact admissible space, nonconstant & duplicate rejection.
toy={x for x in itertools.product(range(3),repeat=3) if len(set(x))>1}
assert all(tuple(decode(encode(x,3),3,3))==x for x in itertools.product(range(3),repeat=3))
a,meta=sample('toy',3,24,['a','b','c']);assert len(a)==24 and {tuple('abc'.index(z) for z in x['trajectory']) for x in a}==toy and meta['space_exhausted']
b,mb=sample('toy',3,23,['a','b','c']);assert len(b)==len({x['trajectory_hash'] for x in b})==23 and mb['rejected_duplicate']>0 and mb['rejected_constant']>0
raw=render(inputs);assert raw==render(list(reversed(inputs)))==render(inputs)
planned=[json.loads(l) for l in raw.splitlines()];by=collections.defaultdict(list)
for x in planned:by[x['sample_id']].append(x)
for inp in inputs:
 xs=by[inp['sample_id']];assert len(xs)==min(inp['B_i'],7**inp['T']-7)==len({x['trajectory_hash'] for x in xs});assert all(len(set(x['trajectory']))>1 and len(x['trajectory'])==inp['T'] for x in xs)
tasks={x['sample_id']:x['task'] for x in co}
for x in planned:x['task']=tasks[x['sample_id']]
raw=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in planned);(R/'RANDOM_PLAN.jsonl').write_text(raw);tsv('RANDOM_PLAN.tsv',[{**x,'trajectory':json.dumps(x['trajectory'])} for x in planned]);planhash=sha(R/'RANDOM_PLAN.jsonl');(R/'RANDOM_PLAN_SHA256.txt').write_text(planhash+'\n')
shutil.copytree(B/'runtime',R/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
protocol=dict(FORMAL_DYNAMIC_RUN=[str(x) for x in roots],MERGED_AUTHORITY=str(M),DYNAMIC_STATUS='COMPLETE',CHECKPOINT_IDENTITY=p['model_sha256'],EXECUTOR_VERSION='V2 write-only unified path',EXECUTOR_SHA256=p['executor_sha256'],CONTEXT=16384,CHUNK=1024,ACTION_SET=ACTIONS,CANDIDATE_LAYERS=p['layers'],TTT_LR=1.0,DELTA_CLIP=1e-5,UPDATE='apply-then-update',TAIL_UPDATE=False,GENERATION_UPDATE=False,DTYPE='bfloat16',ATTENTION_BACKEND='sdpa',DECODING='greedy',SEARCH_BEAM=4,SEARCH_SWEEPS=1,SEARCH_POSITION_ORDER='forward',SCORER_SHA256=sha(B/'runtime/scorer.py'),model_path='/home/zonghan/ttt/models/qwen3_1p7b_stage2',formal_bundle=str(B),expected_versions=p['expected_versions'],executor_rng_seed=p['seed'],hardware='RTX 5090; two workers CPU8-15 and16-23,8 torch threads each; formal remote also RTX5090',FORMAL_SEARCHED_N=1433,STRICT_IMPROVED_N=len(co),EQUAL_N=ct['eq'],WORSE_N=0,RANDOM_REPLICATE_N=1,MASTER_RANDOM_SEED=20260923,RANDOM_REPLICATE_ID=0,seed_derivation='SHA256(20260923|replicate=0|<sample_id>), first8 bytes unsigned big endian',numpy_version=np.__version__,bit_generator='PCG64',sampling='uniform base7 integer; reject only constants and previously sampled; Dynamic overlap allowed',budget_definition='unique fresh-evaluated nonconstant trajectories; excludes7constants and duplicate winner verification',total_budget=sum(x['B_i'] for x in inputs),RANDOM_PLAN_SHA256=planhash,tie_break='shared Fixed wins ties; otherwise earliest frozen random trajectory index',bootstrap=dict(draws=20000,seed=20260923,ordering='tasks lexicographic, samples lexicographic within task',method='paired task-stratified percentile, sample weighted',claim='sample uncertainty conditional on one fixed realization; not seed uncertainty'),no_early_stop=True,no_automatic_retry=True,cohort_claim='outcome-conditioned Dynamic-improved only; not full searched population',preflight_at=datetime.datetime.now().astimezone().isoformat())
put('RANDOM_PROTOCOL.json',protocol);(R/'RANDOM_PROTOCOL.md').write_text('# Single-replicate equal-budget uniform random baseline\n\nOutcome-conditioned final V2 Dynamic-improved cohort only. One pre-frozen realization; no seed-uncertainty or full-population claim.\n\n'+json.dumps(protocol,indent=2)+'\n')
put('CPU_PREFLIGHT.json',dict(status='PASS',P1_bijection_and_rejections=True,P2_budgets=True,P3_sampler_module_reads_no_rewards=True,P4_twice_and_reordered_byte_identical=True,P5_authority_hashes_unchanged=all(sha(Path(n))==h for n,h in authority.items()),formal_search_complete=True,plan_frozen_before_GPU=True,samples=len(co),trajectories=len(planned),toy_rejections=mb,at=datetime.datetime.now().astimezone().isoformat()))
print('CPU_PREFLIGHT_PASS',len(co),len(planned),planhash)
