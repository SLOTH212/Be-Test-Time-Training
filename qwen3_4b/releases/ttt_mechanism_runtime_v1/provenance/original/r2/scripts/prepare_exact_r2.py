#!/usr/bin/env python3
"""Pre-outcome authority, preregistration, manifest, and mapping freeze."""
from __future__ import annotations
import collections, glob, hashlib, itertools, json, os, statistics, time
from datetime import datetime
from pathlib import Path

RUN=Path('/home/USER/ttt/runs/formal/exact_r2_1p7b_v1/run_20260906T013721+0800')
SCREEN=Path('/home/USER/ttt/runs/formal/r2_screening_1p7b_v1/run_20260906T012743+0800')
MODEL=Path('/home/USER/ttt/models/qwen3_1p7b_stage2')
BENCH=Path('/home/USER/ttt/benchmarks/ruler_16k_standard_13task500_v1')
FIXED=Path('/home/USER/ttt/runs/formal/fixed7_1p7b_v1/run_20260828T012948+0800')
DYNAMIC=Path('/home/USER/ttt/runs/formal/dynamic_nonceiling_12task_1p7b_v1/run_20260828T161330+0800')
EXECUTOR=Path('/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/code/dynamic_runner.py')
SCORER=Path('/home/USER/ttt/code/ruler/tools/common.py')
PYTHON='/home/USER/conda_envs/ttt_phase_c_v1/bin/python'
TMUX='/home/USER/ttt_dynamic_mining_v1/phase_d_enriched_dynamic_mining_v1/runtime/tools/tmux'
MODEL_SHA='ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f'
CKPT_SHA='f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd'
BENCH_SHA='5d16e3a6963d4c21450f0530af743730b91bf7b2442699e288e10754f77ad0c5'
RULE_SHA='afeb7f0ceb42cfaa753ea04d926866ba346f83015be92ff8ad5138d5c32f8378'
ELIG_SHA='f966cbfd65f8f9fbc0d079e44cd5e46fb24ca7da72130f5200c84be6faee3a42'
CAND_SHA='737a5244a561e103c660ae33aff7a94e9c53e195d1fc05fc1006a1a7b4979739'
DYNAMIC_SHA='89e5b7da8b85ce063af15216ef9a16de5c433d4e489a19efaae20ad16099689f'
FIXED_SHA='b2c7f7105bdafa53fcebc11815b37fcb1fd3a2ed7005ca749908f754b95b1885'
EXECUTOR_SHA='293886005cb8f78c943f0affe3e3b383d6de238f4bc825a5aea28a6e6dc66f6e'
SCORER_SHA='cee3f5cb3cad1709f81abe3dc32b364ca97322e7ebd91747a8b119aecb0ce556'
ACTIONS=['OFF','L0','L6','L12','L18','L24','ALL']

def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def rows(p):
 with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def atomic_text(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.{time.time_ns()}.tmp');fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
 except BaseException:
  try:os.unlink(t)
  except FileNotFoundError:pass
  raise
def atomic(p,o):atomic_text(p,json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def jsonl(p,z):atomic_text(p,''.join(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for x in z))

rule=SCREEN/'manifests/EXACT_R2_RULE_V1.json';elig=SCREEN/'manifests/EXACT_R2_ELIGIBILITY_V1_MANIFEST.json';cand=SCREEN/'manifests/EXACT_R2_CANDIDATE_MANIFEST.jsonl';parent_path=SCREEN/'census/R2_PRIMARY_PARENT.jsonl'
fixed_audit=FIXED/'audits/FINAL_FIXED7_FORMAL_AUDIT.json';dynamic_audit=DYNAMIC/'audits/FINAL_DYNAMIC_NONCEILING_12TASK_AUDIT.json'
checks={'model':sha(MODEL/'model.safetensors')==MODEL_SHA,'benchmark':sha(BENCH/'benchmark_manifest.json')==BENCH_SHA,
 'rule':sha(rule)==RULE_SHA,'eligibility':sha(elig)==ELIG_SHA,'candidate':sha(cand)==CAND_SHA,
 'dynamic':sha(dynamic_audit)==DYNAMIC_SHA,'fixed7':sha(fixed_audit)==FIXED_SHA,
 'executor':sha(EXECUTOR)==EXECUTOR_SHA,'scorer':sha(SCORER)==SCORER_SHA}
if not all(checks.values()):raise RuntimeError('AUTHORITY_MISMATCH '+json.dumps(checks,sort_keys=True))
if any((RUN/'branches').rglob('*.json')):raise RuntimeError('FORMAL_OUTCOMES_EXIST_BEFORE_PREREGISTRATION')

parent=rows(parent_path);manifest=rows(cand);eligibility=rows(SCREEN/'manifests/EXACT_R2_ELIGIBILITY_V1.jsonl')
if len(parent)!=173 or len(eligibility)!=173 or not all(x['eligible'] for x in eligibility):raise RuntimeError('COHORT_SCOPE')
ids=[x['candidate_id'] for x in manifest];sample_ids={x['canonical_sample_id'] for x in manifest};pairs={(x['canonical_sample_id'],x['canonical_schedule_hash']):x for x in manifest}
by_sample=collections.Counter(x['canonical_sample_id'] for x in manifest);counts=sorted(by_sample.values())
manifest_audit={'status':'PASS','TOTAL_MANIFEST_ROWS':len(manifest),'UNIQUE_CANDIDATE_IDS':len(set(ids)),'SAMPLE_COUNT':len(sample_ids),
 'candidates_per_sample':{'min':min(counts),'median':statistics.median(counts),'mean':statistics.mean(counts),'max':max(counts)},
 'unique_sample_schedule_hashes':len(pairs),'manifest_order_authoritative':True,'candidate_manifest_sha256':sha(cand),'timestamp':now()}
if not(len(manifest)==len(set(ids))==len(pairs)==99120 and len(sample_ids)==173 and min(counts)==462 and statistics.median(counts)==588 and max(counts)==588):raise RuntimeError('MANIFEST_AUDIT_FAILED')
atomic(RUN/'audit/PRELAUNCH_CANDIDATE_MANIFEST_AUDIT.json',manifest_audit)

dyn={}
for p in glob.glob(str(DYNAMIC/'results/*.json')):
 x=json.load(open(p));sid=x.get('sample_id')
 if sid in {q['canonical_sample_id'] for q in parent}:dyn[sid]=x
mapping=[]
for p in parent:
 if p['Dynamic_switch_count']!=1:continue
 seq=p['frozen_Dynamic_action_sequence'];h=canonical(seq);candidate=pairs.get((p['canonical_sample_id'],h))
 if candidate is None:raise RuntimeError('ONE_SWITCH_CANDIDATE_MISSING '+p['canonical_sample_id'])
 x=dyn[p['canonical_sample_id']];pred=x.get('authority_search_result',{}).get('best_prediction_hash')
 if not pred:raise RuntimeError('DYNAMIC_PREDICTION_HASH_MISSING '+p['canonical_sample_id'])
 mapping.append({'canonical_sample_id':p['canonical_sample_id'],'task':p['task'],'benchmark_global_index':p['benchmark_global_index'],
  'dynamic_A':candidate['action_A'],'dynamic_B':candidate['action_B'],'dynamic_tau':candidate['tau'],'T':candidate['T'],
  'dynamic_schedule_hash':h,'candidate_id':candidate['candidate_id'],'frozen_dynamic_score':p['Dynamic_score'],
  'frozen_dynamic_prediction_sha256':pred,'mapping_exact':True})
if len(mapping)!=100:raise RuntimeError(f'ONE_SWITCH_MAPPING_N={len(mapping)}')
mapping_path=RUN/'manifests/ONE_SWITCH_DYNAMIC_TO_R2_CANDIDATE.jsonl';jsonl(mapping_path,mapping)
atomic(RUN/'audit/ONE_SWITCH_DYNAMIC_CANDIDATE_MAPPING_AUDIT.json',{'status':'PASS','ONE_SWITCH_DYNAMIC_CANDIDATE_MAPPING_N':100,
 'mapping_manifest':str(mapping_path),'mapping_manifest_sha256':sha(mapping_path),'all_schedule_hashes_exact':True,'timestamp':now()})

# Predeclare three parity anchors with distinct task, transition pair, and tau.
smoke=None
for combo in itertools.combinations(mapping,3):
 if len({x['task'] for x in combo})==3 and len({(x['dynamic_A'],x['dynamic_B']) for x in combo})==3 and len({x['dynamic_tau'] for x in combo})==3:
  smoke=list(combo);break
if smoke is None:raise RuntimeError('DIVERSE_ONE_SWITCH_SMOKE_NOT_FOUND')
multi=next(x for x in parent if x['Dynamic_switch_count']>1);multi_candidates=[x for x in manifest if x['canonical_sample_id']==multi['canonical_sample_id']]
det={'canonical_sample_id':multi['canonical_sample_id'],'task':multi['task'],'candidate_id':multi_candidates[0]['candidate_id'],
 'intervening_candidate_id':multi_candidates[1]['candidate_id'],'predeclared_before_outcomes':True}

protocol={'schema':'EXACT_R2_1P7B_FORMAL_V1','frozen_at':now(),'primary_comparison':'OFF_VS_R2_VS_DYNAMIC','N':173,'candidates':99120,
 'P1_FORMAL_COMPARATOR':False,'FROZEN_EXPERT_FORMAL_COMPARATOR':False,
 'objective':'same-sample frozen terminal RULER task reward','oracle_type':'offline hindsight structural oracle',
 'tie_break':{'score_tolerance':1e-12,'order':['maximum score','earliest tau','action order A','action order B','sequence SHA256'],'action_order':ACTIONS},
 'primary_table_order':['OFF','R2','Dynamic'],'primary_gain_retention':'SUM(R2-OFF)/SUM(Dynamic-OFF); ratio of sums; raw; unclamped',
 'primary_dynamic_r2_gap':'MEAN(Dynamic-R2)','paired_classification_tolerance':1e-12,
 'secondary_sample_best':True,'predefined_multi_switch_analysis':True,'one_switch_containment_invariant':True,
 'standard_deviation':'population standard deviation (ddof=0)',
 'bootstrap':{'paired_resamples':20000,'paired_seed':20260906,'task_stratified_resamples':20000,'task_stratified_seed':20260907,
  'interval':'2.5th and 97.5th percentiles','retention_resampling_formula':'ratio of sums inside each cohort resample'},
 'execution':{'workers':1,'cpu_affinity':'8-23','manifest_order':True,'clean_candidate_reset':True,'prefix_cache':False,
  'stored_delta_stitching':False,'beam_search':False,'heuristic_pruning':False,'early_stop':False,'atomic_candidate_commit':True},
 'one_switch_smoke_candidate_ids':[x['candidate_id'] for x in smoke],'determinism_smoke':det,
 'claim_boundary':{'tests':'structural complexity','does_not_test':['online predictability','deployability','router quality','observability','decision difficulty']},
 'next_experiment_launch':False}
protocol_path=RUN/'preregistration/EXACT_R2_1P7B_FORMAL_V1.json';atomic(protocol_path,protocol);protocol_sha=sha(protocol_path)
md=f'''# Qwen3-1.7B Formal Exact R2 V1

Frozen before all formal outcomes. Primary comparison: **OFF vs Exact R2 vs Dynamic**. N=173; exact candidate replays=99,120.

Exact R2 is `A^[0:tau) -> B^[tau:T)`, `A != B`, `tau=1..T-1`, using `{ACTIONS}`. It maximizes same-sample frozen terminal RULER reward and is therefore an offline hindsight structural oracle, not an online controller.

Tie-break: maximum score (tolerance 1e-12), earliest tau, action order A, action order B, sequence SHA256. Primary retention is `SUM(R2-OFF)/SUM(Dynamic-OFF)`, raw and unclamped. Paired and task-stratified bootstraps each use 20,000 resamples with seeds 20260906 and 20260907.

P1 and Frozen Expert are excluded. No beam search, pruning, prefix reuse, delta stitching, or early stopping is permitted.
'''
atomic_text(RUN/'preregistration/EXACT_R2_1P7B_FORMAL_V1.md',md)

authority={'schema':'FORMAL_EXECUTION_AUTHORITY','model':{'path':str(MODEL),'model_identity':MODEL_SHA,'checkpoint_identity':CKPT_SHA},
 'benchmark':{'path':str(BENCH),'manifest_sha256':BENCH_SHA},'rule_manifest':{'path':str(rule),'sha256':RULE_SHA},
 'eligibility_manifest':{'path':str(elig),'sha256':ELIG_SHA},'candidate_manifest':{'path':str(cand),'sha256':CAND_SHA},
 'parent_manifest':{'path':str(parent_path),'sha256':sha(parent_path)},'OFF_and_SampleBest':{'root':str(FIXED),'final_audit':str(fixed_audit),'sha256':FIXED_SHA},
 'Dynamic':{'root':str(DYNAMIC),'final_audit':str(dynamic_audit),'sha256':DYNAMIC_SHA},
 'executor':{'path':str(EXECUTOR),'sha256':EXECUTOR_SHA},'scorer':{'path':str(SCORER),'sha256':SCORER_SHA},
 'protocol':{'path':str(protocol_path),'sha256':protocol_sha},'created_at':now()}
authority_path=RUN/'manifests/FORMAL_EXECUTION_AUTHORITY.json';atomic(authority_path,authority)

worker=RUN/'scripts/exact_r2_worker.py';aggregator=RUN/'scripts/aggregate_exact_r2.py';controller=RUN/'scripts/run_controller.sh'
config={'version':'EXACT_R2_1P7B_FORMAL_V1','run_root':str(RUN),'python':PYTHON,'tmux':TMUX,'tmux_session':'exact_r2_1p7b_v1',
 'model_root':str(MODEL),'model_identity':MODEL_SHA,'checkpoint_identity':CKPT_SHA,'benchmark_root':str(BENCH),'benchmark_identity':BENCH_SHA,
 'fixed7_root':str(FIXED),'dynamic_root':str(DYNAMIC),'dynamic_final_audit':str(dynamic_audit),
 'rule_manifest':str(rule),'rule_sha256':RULE_SHA,'eligibility_manifest':str(elig),'eligibility_sha256':ELIG_SHA,
 'candidate_manifest':str(cand),'candidate_manifest_sha256':CAND_SHA,'parent_manifest':str(parent_path),
 'one_switch_mapping':str(mapping_path),'one_switch_mapping_sha256':sha(mapping_path),'protocol':str(protocol_path),'protocol_sha256':protocol_sha,
 'execution_authority_sha256':sha(authority_path),'frozen_executor':str(EXECUTOR),'scorer':str(SCORER),
 'worker':str(worker),'worker_sha256':sha(worker),'aggregator':str(aggregator),'aggregator_sha256':sha(aggregator),
 'controller':str(controller),'controller_sha256':sha(controller),'one_switch_smoke_candidate_ids':protocol['one_switch_smoke_candidate_ids'],
 'determinism_smoke':det,'paired_bootstrap_seed':20260906,'task_stratified_bootstrap_seed':20260907,
 'bootstrap_resamples':20000,'workers':1,'cpu_affinity':'8-23','expected_candidates':99120,'expected_samples':173}
config_path=RUN/'config/EXACT_R2_FORMAL_CONFIG.json';atomic(config_path,config)
preflight={'status':'PASS','authority_checks':checks,'MODEL_IDENTITY_MATCH':True,'CHECKPOINT_IDENTITY_MATCH':True,
 'BENCHMARK_IDENTITY_MATCH':True,'RULE_HASH_MATCH':True,'ELIGIBILITY_HASH_MATCH':True,'CANDIDATE_MANIFEST_HASH_MATCH':True,
 'PROTOCOL_HASH_MATCH':True,'EXECUTION_COHORT_N':173,'TOTAL_CANDIDATES_EXPECTED':99120,'ONE_SWITCH_DYNAMIC_N':100,
 'MULTI_SWITCH_DYNAMIC_N':73,'ONE_SWITCH_DYNAMIC_CANDIDATE_MAPPING_N':100,'P1_FORMAL_COMPARATOR':False,
 'FROZEN_EXPERT_FORMAL_COMPARATOR':False,'formal_outcomes_before_freeze':0,'timestamp':now()}
atomic(RUN/'audit/PRELAUNCH_AUTHORITY_AUDIT.json',preflight)
atomic(RUN/'status/pipeline_status.json',{'pipeline_status':'PREREGISTERED','current_phase':'READY_FOR_SMOKE','formal_launched':False,
 'candidates_committed':0,'candidates_expected':99120,'samples_fully_searched':0,'samples_expected':173,'winners_finalized':0,
 'worker_pid':None,'worker_running':False,'controller_pid':None,'last_error':'NONE','updated_at':now()})
print(json.dumps({'status':'PASS','run_root':str(RUN),'protocol_sha256':protocol_sha,'mapping_n':len(mapping),
 'smoke_candidate_ids':protocol['one_switch_smoke_candidate_ids'],'determinism_smoke':det},indent=2))
