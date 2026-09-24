import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,csv,io,hashlib,math,ast,subprocess,collections,datetime,difflib
import numpy as np
O=Path(__file__).resolve().parent;B=Path('/home/zonghan/ttt')
F=B/'runs/formal/fixed7_old1p7b_chunk1024_v2_reuse_20260918';D=B/'transfer/dynamic_old1k_v2_noncwe_half_20260918';M=B/'runs/formal/mechanism_old1k_v2_20260921';A=B/'audits/dynamic_old1k_v2_merge_20260921';E=B/'audits/latest_dynamic_mechanism_cpu_20260923';V=B/'audits/reference_executor_v2_stage1_20260915';V2=B/'audits/reference_executor_v2_stage2_1435_20260915';L=B/'runs/formal/dynamic_old1k_v2_noncwe_local_20260918';R=A/'dynamic_old1k_v2_remote_transfer_20260921/dynamic_old1k_v2_remote_results'
prov={}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p).resolve();s=p.stat();q=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,mtime=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),sha256=digest(p))
 if str(p) in prov:assert prov[str(p)]==q,str(p)
 prov[str(p)]=q;return q['sha256']
def read(p):record(p);return Path(p).read_text()
def j(p):return json.loads(read(p))
def cs(p):return list(csv.DictReader(io.StringIO(read(p))))
def jl(p):return [json.loads(s) for s in read(p).splitlines() if s.strip()]
def put(name,x): (O/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def tsv(name,rs):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]),delimiter='\t');w.writeheader();w.writerows(rs)
def close(a,b,tol=1e-12):assert abs(a-b)<=tol,(a,b)
def mean(v):return float(np.mean(v))
ACT=['OFF','L0','L6','L12','L18','L24','ALL'];CWE='ruler_cwe_16k';TOL=1e-12
fp=j(F/'PROTOCOL.json');dp=j(D/'PROTOCOL.json');mp=j(M/'PROTOCOL.json')
for root,names in [(F,['COMPLETE.json','SUMMARY.json','BRIDGE_0.json','BRIDGE_1.json','run.py']), (D,['pipeline.py','MANIFEST.json']), (L,['COMPLETE.json','RUN_CONFIG.json','bridge_0.json','bridge_1.json']), (R,['COMPLETE.json','RUN_CONFIG.json','bridge_0.json','bridge_1.json']), (M,['COMPLETE.json','SOURCE_HASHES.json','INSTRUMENTATION.diff','SMOKE_0.json','SMOKE_1.json','run.py','summarize.py']), (E,['CPU_AUDIT_REPORT.md','summary.json','supplementary_summary.json','integrity.json','audit.py','report.py','source_preservation.json','independent_crosscheck.json','input_provenance.json']), (E/'completion',['COMPLETION_REPORT.md','FINAL_STATUS.json','numerics.py','source_preservation.json','checks.json','independent_checks.json','input_provenance.json']), (V,['V2_PRECHECK_RESULTS.json','V2_PRECHECK_PROTOCOL.json','V2_STAGE1_SUMMARY.json','V2_SOURCE_SHA256.json','V2_DESIGN.md','REFERENCE_EXECUTOR_V2_DESIGN.md','V2_FORMAL_COMPLETION.json','V2_INSTRUMENTATION_BRIDGE.json','V2_POSTFLIGHT_PACKAGE_AUDIT.json','precheck.py','verify_package.py']), (V2,['FROZEN_V2_STAGE2_PROTOCOL.json','V2_STAGE2_COMPLETION.json','V2_STAGE2_POSTFLIGHT.json'])]:
 for n in names:
  p=root/n
  if p.exists():record(p)
assert fp['actions']==dp['actions']==ACT
assert fp['model_sha256']==dp['model_sha256']==mp['model_sha256']
assert fp['chunk']==dp['chunk_size']==mp['chunk']==1024
assert record(D/'runtime/executor.py')==dp['executor_sha256']
assert record(F/'FIXED7_FULL6500.csv')==dp['fixed_source_sha256']
assert record(D/'PROTOCOL.json')==mp['source_protocol_sha256']
for rel,h in fp['source_hashes'].items():assert record(F/rel)==h
for rel,h in dp['inputs'].items():assert record(D/rel)==h
for p in (D/'runtime').rglob('*.py'):record(p)
for rel,h in j(M/'SOURCE_HASHES.json').items():assert record(M/rel)==h
for rel,h in j(D/'MANIFEST.json').items():
 if rel!='model/model.safetensors':assert record(D/rel)==h
assert j(D/'MANIFEST.json')['model/model.safetensors']==dp['model_sha256']
# Weight file is deliberately not opened or loaded; identity is cross-verified from frozen protocols and per-result hashes.
benchmark=jl(Path(fp['benchmark']));assert record(Path(fp['benchmark']))==fp['benchmark_sha256']
ids=[r['sample_id'] for r in benchmark];assert len(ids)==len(set(ids))==6500
bt={r['sample_id']:r['task'] for r in benchmark};assert len(set(bt.values()))==13 and set(collections.Counter(bt.values()).values())=={500}
fixed={};cellmeta={}
for r in cs(F/'FIXED7_FULL6500.csv'):
 sid,a=r['sample_id'],r['action'];assert sid in bt and bt[sid]==r['task'] and a in ACT
 key=(sid,a);assert key not in cellmeta
 sc=float(r['score']);assert math.isfinite(sc) and 0<=sc<=1
 cellmeta[key]=r;fixed.setdefault(sid,{})[a]=sc
assert set(fixed)==set(ids) and len(cellmeta)==45500 and all(set(v)==set(ACT) for v in fixed.values())
# Verify the merged raw fixed table against all underlying current V2 record files.
seen=set();raw_new=0
for p in sorted((F/'new_samples').glob('*.json')):
 x=j(p);sid=x['sample_id'];assert sid not in seen;seen.add(sid)
 assert x['model_sha256']==fp['model_sha256'] and x['protocol_sha256']==record(F/'PROTOCOL.json') and x['checkpoint_unchanged']
 assert set(x['fixed'])==set(ACT)
 for a,r in x['fixed'].items():
  q=cellmeta[sid,a];assert q['source']=='new_missing_v2';assert float(q['score'])==r['score'] and q['prediction_hash']==r['prediction_hash'] and q['input_token_sha256']==x['input_token_sha256'];raw_new+=1
reuse=cs(F/'reused/V2_FIXED7_1435.csv');assert record(F/'reused/V2_FIXED7_1435.csv')==fp['reused_sha256'];reusekeys=set()
for r in reuse:
 sid,a=r['sample_id'],r['action'];key=sid,a;assert key not in reusekeys;reusekeys.add(key);assert sid not in seen
 q=cellmeta[key]
 # Original Stage-2 fixed columns are inspected and matched directly.
 sc=float(r.get('score',r.get('reward')));assert q['source']=='reused_v2_1435';close(float(q['score']),sc)
 assert q['prediction_hash']==r['prediction_hash']
assert raw_new==35455 and len(reusekeys)==10045
sb={sid:max(x.values()) for sid,x in fixed.items()}
for r in cs(F/'SAMPLE_BEST_FULL6500.csv'):close(float(r['sample_best']),sb[r['sample_id']])
order=jl(D/'data/fixed_all6500.jsonl');assert [r['sample_id'] for r in order]==ids
for r in order:assert r['task']==bt[r['sample_id']];close(r['sample_best'],sb[r['sample_id']])
fixedtable=[]
for t in sorted(set(bt.values())):
 si=[sid for sid in ids if bt[sid]==t];fixedtable.append(dict(task=t,N=len(si),**{a:mean([fixed[s][a] for s in si])*100 for a in ACT},SampleBest=mean([sb[s] for s in si])*100))
tsv('TABLE_FIXED13_V2.tsv',fixedtable)
print('FIXED_COMPLETE: 6500 samples /45500 source-verified cells',flush=True)
sources=j(A/'RESULT_SOURCES.json');records={};candidate_total=0
for machine,root in [('local',L),('remote',R)]:
 expected=set(j(D/f'data/assignment_{machine}.json'));got=set()
 for p in sorted((root/'results').glob('*.json')):
  x=j(p);sid=x['sample_id'];assert sid not in records and sid in expected;got.add(sid)
  assert record(p)==sources[sid]['sha256'];assert Path(sources[sid]['path']).resolve()==p.resolve()
  assert x['protocol_hash']==record(D/'PROTOCOL.json') and x['model_sha256']==dp['model_sha256'] and x['executor_sha256']==dp['executor_sha256']
  assert x['fixed_reference_match'] and x['fixed_scores']==fixed[sid];close(x['sample_best'],sb[sid]);close(x['gain'],x['dynamic_score']-sb[sid]);assert x['dynamic_score']>=sb[sid]
  w=x['winner'];assert w['score']==x['dynamic_score'] and w['ttt_chunk_size']==1024 and w['checkpoint_hash']==dp['model_sha256'];assert w['clean_full_replay'] and w['recompute_downstream_updates'] and not w['stored_delta_stitching'] and w['generation_update_count']==0
  assert len(x['candidates'])==x['candidate_count'] and max(z['score'] for z in x['candidates'])==w['score']
  assert any(z['sequence']==w['sequence'] and z['prediction_hash']==w['prediction_hash'] for z in x['candidates'])
  candidate_total+=len(x['candidates']);records[sid]=x
 assert got==expected
assert set(records)=={s for s in ids if bt[s]!=CWE and sb[s]<1}
rows=[]
for sid in ids:
 t=bt[sid];measured=sid in records;seq=records[sid]['dynamic_score'] if measured else sb[sid]
 rows.append(dict(sample_id=sid,task=t,sample_best=sb[sid],sequential_score=seq,sequential_minus_sb=seq-sb[sid],source='CWE_SB_FALLBACK' if t==CWE else 'V2_DYNAMIC',dynamic_measured='YES' if measured else 'NO',static_ceiling_carried=t!=CWE and not measured))
tsv('FULL13_SEQLB_SAMPLE_LEVEL.tsv',rows)
# Extract only the pure authoritative CI function via AST. Do not execute its V1 comparison or file-writing top level.
authtext=read(A/'audit_merge.py');tree=ast.parse(authtext);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ci');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(A/'audit_merge.py'),'exec'),ns);ci=ns['ci']
def stats(rs,with_ci=True):
 z=dict(N=len(rs),sample_best=mean([r['sample_best'] for r in rs])*100,sequential_score=mean([r['sequential_score'] for r in rs])*100,gain_pp=mean([r['sequential_minus_sb'] for r in rs])*100,improved_n=sum(r['sequential_minus_sb']>TOL for r in rs),equal_n=sum(abs(r['sequential_minus_sb'])<=TOL for r in rs),worse_n=sum(r['sequential_minus_sb']< -TOL for r in rs))
 if with_ci:z['gain_ci95_pp']=ci([dict(task=r['task'],gain=r['sequential_minus_sb']) for r in rs])
 return z
r12=[r for r in rows if r['task']!=CWE];searched=[r for r in r12 if r['dynamic_measured']=='YES'];improved=[r for r in searched if r['sequential_minus_sb']>TOL]
main=stats(r12);main.update(task_n=len({r['task'] for r in r12}),ceiling_n=len(r12)-len(searched),searched=stats(searched),improved_only={**stats(improved,False),'OUTCOME_CONDITIONED':'YES'})
assert main['searched']['improved_n']+main['searched']['equal_n']+main['searched']['worse_n']==len(searched) and main['worse_n']==0
previous=j(A/'FINAL_AUDIT.json');assert main['gain_ci95_pp']==previous['bootstrap']['full6000_gain_CI'] and main['searched']['gain_ci95_pp']==previous['bootstrap']['searched1433_gain_CI']
for a,b in [('sample_best','sample_best'),('sequential_score','dynamic'),('gain_pp','gain_pp')]:close(main[a],previous['full6000'][b])
full=stats(rows);full['composition']='12-task V2 Dynamic population + CWE V2 Sample Best fallback';close(full['gain_pp'],len(r12)/len(rows)*main['gain_pp']);full['identity_abs_error']=abs(full['gain_pp']-len(r12)/len(rows)*main['gain_pp'])
cw=[r for r in rows if r['task']==CWE];cwe=dict(N=len(cw),OFF=mean([fixed[r['sample_id']]['OFF'] for r in cw])*100,SampleBest=mean([r['sample_best'] for r in cw])*100,DYNAMIC_MEASURED='NO',INCREMENTAL_CREDIT='ZERO_BY_FALLBACK',measured_dynamic_effect='NOT_AVAILABLE')
bootstrap=dict(authority_script=str(A/'audit_merge.py'),authority_script_sha256=record(A/'audit_merge.py'),pure_function_extraction='AST ci function only; no execution of file-writing or historical comparison code',seed=20260921,RNG='numpy.default_rng / PCG64',numpy=np.__version__,draw_count=20000,batch_size=1000,quantile='numpy.quantile([.025,.975]), linear',task_order=sorted(set(bt.values())),sample_order_source=str(D/'data/fixed_all6500.jsonl'),sample_order_sha256=record(D/'data/fixed_all6500.jsonl'),pairing='per-sample sequential_score minus current V2 fixed maximum',cwe='500 zero deltas assigned by fallback; RNG draws retained',DYN12_CI_REPRODUCED_EXACT=True,SEARCHED_CI_REPRODUCED_EXACT=True,FULL13_GAIN_CI=full['gain_ci95_pp'],full13_status='NEW_CPU_CASE_B_COMPOSITE_USING_EXISTING_AUTHORITY_METHOD')
put('FULL13_BOOTSTRAP_AUDIT.json',bootstrap)
tasktable=[]
for t in sorted(set(bt.values())):
 rr=[r for r in rows if r['task']==t];ss=[r for r in rr if r['dynamic_measured']=='YES'];st=stats(rr,False);su=stats(ss,False) if ss else None
 tasktable.append(dict(task=t,N=len(rr),sample_best=st['sample_best'],sequential_score=st['sequential_score'],full_task_gain_pp=st['gain_pp'] if t!=CWE else '0_BY_FALLBACK',dynamic_measured='NO' if t==CWE or not ss else 'YES',searched_n=len(ss) if t!=CWE else 'NA',improved_n=st['improved_n'] if t!=CWE else 'NA',searched_sb=su['sample_best'] if su else 'NA',searched_dynamic=su['sequential_score'] if su else 'NA',searched_gain_pp=su['gain_pp'] if su else 'NA'))
tsv('TABLE_TASK13_V2_SEQLB.tsv',tasktable)
tsv('DYNAMIC_WINNER_PROVENANCE.tsv',[dict(sample_id=sid,task=bt[sid],source_path=sources[sid]['path'],source_sha256=sources[sid]['sha256'],sequence_hash=x['winner']['sequence_hash'],prediction_hash=x['winner']['prediction_hash'],protocol_sha256=x['protocol_hash'],executor_sha256=x['executor_sha256']) for sid,x in sorted(records.items())])
print('DYNAMIC_COMPLETE',main,'FULL13',full,'CWE',cwe,flush=True)
# Reconstruct mechanism effects directly, preserving official COHORT order and exact bootstrap function.
cohort=j(M/'COHORT.json');assert {c['sample_id'] for c in cohort}=={r['sample_id'] for r in improved} and len(cohort)==170
core={x['sample_id']:x for x in (j(p) for p in sorted((M/'core').glob('*.json')))};r2={x['sample_id']:x for x in (j(p) for p in sorted((M/'r2').glob('*.json')))};assert len(core)==len(r2)==170
for c in cohort:
 sid=c['sample_id'];assert c['dynamic_source']==sources[sid];close(c['sample_best'],sb[sid]);assert c['sequence']==records[sid]['winner']['sequence'];close(c['native_score'],records[sid]['dynamic_score']);assert core[sid]['native_repeat_pass'];assert core[sid]['protocol_sha256']==record(M/'PROTOCOL.json')==r2[sid]['protocol_sha256']
mt=read(M/'summarize.py');fn=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='stat');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(M/'summarize.py'),'exec'),ns);mstat=ns['stat'];ms=j(M/'SUMMARY.json');mechtable=[];mechanism_pairs=[]
for label in ['reverse','reset','nontrivial_reset','deletion','sham','K0','K1','r2']:
 items=[];manifest=[]
 for c in cohort:
  sid=c['sample_id'];x=core[sid];n=c['native_score']
  if label=='r2':v=r2[sid]['r2_score']
  elif label=='reverse':v=x['reverse']['score']
  elif label in ['reset','nontrivial_reset']:
   if not c['reset_eligible'] or (label=='nontrivial_reset' and not c['nontrivial_reset']):continue
   v=x['reset']['score']
  elif label=='deletion':
   if not x['deletion']:continue
   v=sum(z['result']['score'] for z in x['deletion'])/len(x['deletion'])
  elif label=='sham':
   if not x['sham']:continue
   v=x['sham']['score']
  else:
   if not x['K']:continue
   v=x['K'][label]['score']
  items.append((c['task'],n,v));manifest.append(sid);mechanism_pairs.append(dict(experiment=label,sample_id=sid,task=c['task'],sample_best=sb[sid],native=n,condition=v,native_minus_condition=n-v))
 st=mstat(items);assert st==ms['results'][label],(label,st,ms['results'][label]);man='\n'.join(manifest)+'\n';(O/f'COHORT_{label}.txt').write_text(man)
 mechtable.append(dict(experiment=label,N=st['n'],SampleBest_mean_if_available=mean([sb[s] for s in manifest])*100,Native_mean=st['native_percent'],Condition_mean=st['condition_percent'],paired_effect_pp=st['native_minus_condition_pp'],CI_low=st['CI95'][0],CI_high=st['CI95'][1],decrease_n=st['lower'],equal_n=st['equal'],increase_n=st['higher'],cohort_manifest_hash=hashlib.sha256(man.encode()).hexdigest()))
tsv('TABLE_MECHANISM_V2.tsv',mechtable);tsv('MECHANISM_SAMPLE_PAIRS.tsv',mechanism_pairs)
print('MECHANISM_RECOMPUTED_EXACT',flush=True)
# R2 completeness, reuse lineage, multiplicity, timing and count-matched reversal independently checked.
es=j(E/'summary.json');sup=j(E/'supplementary_summary.json');per=cs(E/'per_sample_improved.csv');permap={x['sample_id']:x for x in per};setwise=cs(E/'completion/setwise_summary.csv');corr=cs(E/'gain_complexity_correlations.csv');center=cs(E/'completion/within_task_rank_correlations.csv');directions=cs(E/'completion/cross_task_direction_consistency.csv');dominant=cs(E/'completion/dominant_transition_by_task.csv')
structure_rows=[];totalpairs=totalcand=reuse_n=0
for c in cohort:
 sid=c['sample_id'];x=r2[sid];T=c['T'];candidates=x['candidates'];by={(z['A'],z['B'],z['tau']):z for z in candidates}
 assert len(by)==len(candidates)==42*(T-1)==x['candidate_count'];assert set(by)=={(a,b,t) for a in ACT for b in ACT if a!=b for t in range(1,T)}
 dynby={tuple(z['sequence']):z for z in records[sid]['candidates']}
 for (a,b,t),z in by.items():
  if z['reused']:
   q=dynby[tuple([a]*t+[b]*(T-t))];assert z['score']==q['score'] and z['prediction_hash']==q['prediction_hash'];reuse_n+=1
 assert max(z['score'] for z in candidates)==x['r2_score']
 matched=[z for z in candidates if abs(z['score']-c['native_score'])<=TOL];assert all(z['score']<=c['native_score']+TOL for z in candidates)
 adjacent=any((z['A'],z['B'],z['tau']+1) in by and abs(by[z['A'],z['B'],z['tau']+1]['score']-c['native_score'])<=TOL for z in matched)
 pred=any(z['prediction_hash']==c['native_prediction_hash'] for z in matched)
 revs=[z['score']-by[z['B'],z['A'],T-z['tau']]['score'] for z in matched]
 unord=[]
 for key,z in by.items():
  a,b,t=key;revkey=b,a,T-t
  if key<revkey:unord.append(z['score']-by[revkey]['score'])
 totalpairs+=len(unord);totalcand+=len(candidates)
 rec=(x['r2_score']-sb[sid])/(c['native_score']-sb[sid]);p=permap[sid];close(rec,float(p['r2_recovery']));assert int(p['r2_match_n'])==len(matched) and (p['any_matching_prediction_match']=='True')==pred
 structure_rows.append(dict(sample_id=sid,task=c['task'],switches=c['switch_count'],match_n=len(matched),recovery=rec,adjacent=adjacent,prediction_match=pred,match_reverse_mean=mean(revs) if revs else None,all_matches_strictly_beat_reverse=bool(revs) and all(v>TOL for v in revs),global_unequal_fraction=mean([abs(v)>TOL for v in unord]),global_abs_diff=mean([abs(v) for v in unord])))
assert totalcand==ms['R2']['candidates']==97356 and reuse_n==ms['R2']['reused']==7733
multi=[r for r in structure_rows if r['switches']>1];matchedmulti=[r for r in multi if r['match_n']];matchedall=[r for r in structure_rows if r['match_n']]
allset=next(x for x in setwise if x['set']=='match' and x['scope']=='all');multiset=next(x for x in setwise if x['set']=='match' and x['scope']=='native_multi');rho=next(x for x in corr if x['cohort']=='improved170' and x['metric']=='switch_count')
close(mean([r['match_reverse_mean'] for r in matchedall]),float(allset['reverse_mean_delta']));close(mean([r['match_reverse_mean'] for r in matchedmulti]),float(multiset['reverse_mean_delta']))
ratio=sum(r2[c['sample_id']]['r2_score']-sb[c['sample_id']] for c in cohort)/sum(c['native_score']-sb[c['sample_id']] for c in cohort);close(ratio,sup['all']['recovery_sum_gain_ratio'])
rstats=dict(R2_DYNAMIC_IMPROVED_N=len(cohort),R2_CANDIDATE_N=totalcand,R2_REUSED_N=reuse_n,R2_NATIVE_ONE_SWITCH_N=sum(r['switches']==1 for r in structure_rows),R2_NATIVE_MULTI_SWITCH_N=len(multi),R2_MATCH_ALL_N=len(matchedall),R2_MATCH_MULTI_N=len(matchedmulti),R2_UNMATCHED_MULTI_N=len(multi)-len(matchedmulti),R2_MEAN_PER_SAMPLE_RECOVERY=mean([r['recovery'] for r in structure_rows]),R2_MEAN_RECOVERY_CI=es['improved170']['recovery_ci95'],R2_RATIO_OF_SUMS_RECOVERY=ratio,R2_MULTI_MULTIPLE_MATCH_N=sum(r['match_n']>1 for r in matchedmulti),R2_MULTI_UNIQUE_MATCH_N=sum(r['match_n']==1 for r in matchedmulti),R2_MATCH_COUNT_MEDIAN=float(np.median([r['match_n'] for r in matchedmulti])),R2_ADJACENT_SUCCESS_BOUNDARY_N=sum(r['adjacent'] for r in matchedmulti),R2_ISOLATED_WITHIN_PAIR_N=sum(not r['adjacent'] for r in matchedmulti),R2_MATCH_PREDICTION_HASH_ANY_ALL=sum(r['prediction_match'] for r in matchedall),R2_MATCH_PREDICTION_HASH_ANY_MULTI=sum(r['prediction_match'] for r in matchedmulti),R2_GAIN_SWITCH_SPEARMAN=float(rho['rho']),R2_GAIN_SWITCH_CI=[float(rho['ci_low']),float(rho['ci_high'])],REVERSAL_ALL_MATCHED_MEAN_PP=float(allset['reverse_mean_delta'])*100,REVERSAL_ALL_MATCHED_CI=[float(allset[k])*100 for k in ['reverse_mean_delta_ci_low','reverse_mean_delta_ci_high']],REVERSAL_MULTI_MATCHED_MEAN_PP=float(multiset['reverse_mean_delta'])*100,REVERSAL_MULTI_MATCHED_CI=[float(multiset[k])*100 for k in ['reverse_mean_delta_ci_low','reverse_mean_delta_ci_high']],MULTI_ALL_MATCHES_STRICTLY_BEAT_REVERSE_N=sum(r['all_matches_strictly_beat_reverse'] for r in matchedmulti),GLOBAL_UNORDERED_PAIR_N=totalpairs,GLOBAL_PAIR_SCORE_DIFFERENT_PCT=mean([r['global_unequal_fraction'] for r in structure_rows])*100,GLOBAL_ABS_DIFF_MEAN_PP=mean([r['global_abs_diff'] for r in structure_rows])*100)
close(rstats['R2_MEAN_PER_SAMPLE_RECOVERY'],es['improved170']['metrics']['r2_recovery']['mean']);close(rstats['GLOBAL_ABS_DIFF_MEAN_PP'],es['improved170']['available_reverse_mean_abs_delta']*100)
ts=[]
for k,v in rstats.items():
 scope='170 improved (outcome-conditioned)'
 if k in ['R2_MULTI_MULTIPLE_MATCH_N','R2_MULTI_UNIQUE_MATCH_N','R2_MATCH_COUNT_MEDIAN','R2_ADJACENT_SUCCESS_BOUNDARY_N','R2_ISOLATED_WITHIN_PAIR_N','R2_MATCH_PREDICTION_HASH_ANY_MULTI','REVERSAL_MULTI_MATCHED_MEAN_PP','REVERSAL_MULTI_MATCHED_CI','MULTI_ALL_MATCHES_STRICTLY_BEAT_REVERSE_N']:scope='62 score-matched native-multi-switch samples'
 if k.startswith('REVERSAL_ALL') or k=='R2_MATCH_PREDICTION_HASH_ANY_ALL':scope='154 score-matched samples'
 ts.append(dict(metric=k,value=json.dumps(v),scope=scope,weighting='sample equal; candidates equal within sample' if k.startswith(('GLOBAL_','REVERSAL_')) else 'see R2_STATISTICAL_DEFINITIONS.json',source=str(E)))
tsv('TABLE_R2_STRUCTURE_V2.tsv',ts);tsv('R2_SAMPLE_CROSSCHECK.tsv',structure_rows)
put('R2_STATISTICAL_DEFINITIONS.json',dict(scope='All metrics outcome-conditioned on 170 current V2 improvements unless explicitly subset',match='score equality within 1e-12; prediction equality separately measured',recovery='(R2-SB)/(Native-SB), unclipped; ratio-of-sums differs from mean per-sample recovery',multiplicity_scope='62 matched multi-switch samples, not all78',adjacent='number of samples with at least one adjacent successful tau within the SAME ordered action pair; isolated is remaining matched-multi samples',reversal='A^tau B^(T-tau) compared to B^(T-tau) A^tau; all matched candidates, sample-equal mean; selected-set contrast subject to selection bias',global_weighting='48678 unordered pairs; first mean within sample then across170; not pooled candidate weighting',bootstrap=dict(seed=20260923,mean_draws=4000,correlation_draws=2000,method='task-stratified paired sample bootstrap',sources=[str(E/'audit.py'),str(E/'completion/numerics.py')]),prediction_hash='equality of recorded decoded prediction only, not state equivalence'))
boundaries=dict(NOT_SUPPORTED=['early state formation -> late exploitation','universal shallow -> deep transition','R2 failure caused by revisitation','larger gain requires more switches','broad timing tolerance from multiple optima'],alignment=es['multi_matched']['alignment'],alignment_scope='62 multi-switch matched, all matching candidates equal within sample',setwise_supplement=setwise,matched_revisit_n=es['multi_matched']['revisit_n'],matched_n=62,unmatched_revisit_n=es['multi_unmatched']['revisit_n'],unmatched_n=16,revisit_fisher_two_sided=sup['revisit_fisher_two_sided'],task_centered_correlations=center,cross_task_direction_consistency=directions,dominant_transition_by_task=dominant,additional_limits=['R2 is answer-scored hindsight enumeration on fixed action set and chunk grid.','92 native one-switch cases are structurally easy and may reuse identical current Dynamic records, not independent replay evidence.','Same prediction hash is not hidden-state equivalence.','Reversal comparisons of score-selected sets are not unbiased causal effects.','No explicit replay_index or parent fields; candidate discovery index is only a proxy.','K closure is restricted to the recorded first-continuation tensor sites, suffix hidden/logits and prediction; not historical complete L1-L6.','Random baseline is still running and excluded from this final completed-evidence package.'])
(O/'R2_CLAIM_BOUNDARIES.md').write_text('# R2 CLAIM BOUNDARIES\n\n'+ '\n'.join('NOT_SUPPORTED: '+s for s in boundaries['NOT_SUPPORTED'])+'\n\n'+read(E/'CPU_AUDIT_REPORT.md')+'\n\nSupplementary completed numerical analysis:\n\n'+read(E/'completion/COMPLETION_REPORT.md')+'\n\nMachine-readable exact values and definitions are in WRITER_DATA_HANDOFF.json and R2_STATISTICAL_DEFINITIONS.json.\n')
print('R2_STRUCTURE_VERIFIED',rstats,flush=True)
# Executor audit: distinguish original tested runner, instrumented runner and final adapters.
vsum=j(V/'V2_STAGE1_SUMMARY.json');pre=j(V/'V2_PRECHECK_RESULTS.json');vsha=j(V/'V2_SOURCE_SHA256.json');oldsource=read(V/'source/reference_executor_v2.py');finalsource=read(D/'runtime/executor.py');fixedsource=read(F/'source/executor.py')
assert record(V/'source/reference_executor_v2.py')==vsum['source_sha256']
record(V/'source/instrumented_reference_executor_v2.py');record(V/'source/instrumented_ttt_state_core.py')
valid=[]
def gate(name,N,passes,criterion,source,status=None,scope='original V2 precheck',sha=None):
 valid.append(dict(test_name=name,N=N,pass_n=passes,fail_n=N-passes if isinstance(N,int) and isinstance(passes,int) else 'NOT_AVAILABLE',criterion=criterion,source_report=str(source),status=status or ('PASS_EXACT' if N==passes else 'FAIL'),scope=scope,executor_sha256=sha or vsum['source_sha256']))
for name,file,fields in [('same-path repeatability','same_path_repeat.csv',['trace_exact','prediction_match','reward_match']),('zero-write schedule invariance','zero_write.csv',['trace_exact','prediction_match','reward_match','weight_zero_A','weight_zero_B']),('future-action prefix invariance','future_prefix.csv',['trace_exact']),('apply-then-update','apply_then_update.csv',['entry_exact','current_output_exact'])]:
 rr=cs(V/'precheck'/file);passed=sum(all(r[f]=='True' for f in fields) for r in rr);assert passed==len(rr);gate(name,len(rr),passed,'exact tensor/hash equality at recorded sites; '+','.join(fields),V/'precheck'/file)
isol=cs(V/'precheck/checkpoint_isolation.csv');gate('checkpoint isolation / precheck',len(isol),sum(x['checkpoint_unchanged']=='True' for x in isol),'candidate writable checkpoint tensor digests unchanged (not every model tensor)',V/'precheck/checkpoint_isolation.csv');gate('state isolation / precheck',len(isol),sum(x['clean_state']=='True' for x in isol),'clean state, fresh full replay; no controlled donor fork',V/'precheck/checkpoint_isolation.csv')
for root,file,scope in [(V,'V2_STATE_ISOLATION.csv','Stage-1 173x8'),(V2,'V2_STATE_ISOLATION_1435.csv','Stage-2 1435x8')]:
 rr=cs(root/file)
 for name,field in [('checkpoint isolation','checkpoint_unchanged'),('state isolation','state_reset_pass')]:gate(name+' / '+scope,len(rr),sum(x[field]=='True' for x in rr),'exact recorded digest or fresh evaluator/cache reset checks',root/file,scope=scope)
# Original precheck zero-write traces explicitly cover all5 candidate outputs in full chunks, tail and generation.
tracecoverage=[]
for p in sorted((V/'precheck/pair_metrics').glob('V2_Z_*')):
 x=j(p);T=len(x['run_A']['sequence']);met={z['site']:z for z in x['trace_metrics']};sites={phase:[] for phase in ['full_chunk','tail','generation']}
 for key,z in met.items():
  if key.startswith('mlp_output/'):
   parts=key.split('/');layer=int(parts[1][1:]);chunk=int(parts[2][1:])
   if layer in [0,6,12,18,24]:sites['generation' if chunk<0 else 'tail' if chunk>T else 'full_chunk'].append((layer,z))
 for phase,zz in sites.items():
  present={l for l,z in zz};passed=present=={0,6,12,18,24} and all(z['exact'] for l,z in zz)
  tracecoverage.append(dict(pair=p.name,phase=phase,site_n=len(zz),all_five_present=present=={0,6,12,18,24},all_exact=passed))
for phase in ['full_chunk','tail','generation']:
 rr=[r for r in tracecoverage if r['phase']==phase];gate(phase+' common path / zero-write trace',len(rr),sum(r['all_exact'] for r in rr),'all 5 candidate MLP outputs captured and bitwise equal between zero-write schedules; original instrumented V2',V/'precheck/pair_metrics',scope='original precheck; zero-write functional output equivalence')
tsv('EXECUTOR_TRACE_COVERAGE.tsv',tracecoverage)
# Static source checks are never presented as numerical reruns.
def fnnode(src,name):return next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
oldfn=ast.dump(fnnode(oldsource,'install_dynamic_forward'),include_attributes=False);newfn=ast.dump(fnnode(finalsource,'install_dynamic_forward'),include_attributes=False);fixfn=ast.dump(fnnode(fixedsource,'install_dynamic_forward'),include_attributes=False)
assert oldfn==newfn==fixfn
assert 'V2_CANDIDATE_STANDARD_PATH_FORBIDDEN' in finalsource and 'active=int(self.layer_idx) in ACTION_LAYERS[action]' in finalsource
assert 'if full_len<x.shape[1]:outputs.append(F.linear(h[:,full_len:],state.weight,self.down_proj.bias))' in finalsource
assert 'return F.linear(h,present,self.down_proj.bias),present' in finalsource
for name in ['ACTION_CONTROLS_WRITE_ONLY','ALL_CANDIDATE_LAYERS_COMMON_PATH','FULL_CHUNK_COMMON_PATH','TAIL_COMMON_PATH','GENERATION_COMMON_PATH']:
 gate(name,'NOT_APPLICABLE','NOT_APPLICABLE','static source review and AST-identical install_dynamic_forward to validated original; empirical original precheck reported separately',D/'runtime/executor.py',status='STATIC_VERIFIED',scope='final Dynamic + final Fixed adapters',sha=dp['executor_sha256'])
# Record core update semantics and scorer code placement without importing either.
core_source=read(D/'runtime/code/hf_models/hf_qwen3/ttt_state_core.py');gate('scorer isolation','NOT_FOUND','NOT_FOUND','No V2 formal alternate-reference intervention result located. Static scorer call occurs after generate/decode; this does not replace an empirical isolation test.',D/'runtime/executor.py',status='NOT_FOUND',scope='V2 formal empirical scorer isolation',sha=dp['executor_sha256'])
for root,machine in [(L,'local'),(R,'remote')]:
 for worker in [0,1]:
  x=j(root/f'bridge_{worker}.json');assert x['status']=='PASS';gate('final adapter Fixed7 bridge '+machine+str(worker),x['n']*7,x['n']*7,'7 fixed scores and prediction hashes vs frozen V2 reference per bridge sample; N counts cells',root/f'bridge_{worker}.json',scope='final Dynamic runner; bridge sample×action cells',sha=dp['executor_sha256'])
gate('final Dynamic winner replay parity',len(records),len(records),'winner score/prediction exactly present in candidate pool; formal driver force-replay parity gate and completion',A/'RESULT_SOURCES.json',scope='final Dynamic selected winners',sha=dp['executor_sha256'])
gate('final mechanism Native replay parity',len(core),sum(x['native_repeat_pass'] for x in core.values()),'recorded native score and prediction match current Dynamic winner',M/'core',scope='instrumented mechanism adapter',sha=record(M/'runtime/executor.py'))
gate('full precheck rerun on final adapter hash','NOT_FOUND','NOT_FOUND','Original prechecks + AST identical functional forward + final score/prediction bridges exist; full original precheck suite not rerun under final adapter hash',D/'runtime/executor.py',status='NOT_FOUND',scope='final adapter only',sha=dp['executor_sha256'])
tsv('TABLE_V2_EXECUTOR_VALIDATION.tsv',valid)
lineage=dict(original_validated_executor_sha256=vsum['source_sha256'],precheck_instrumented_executor_sha256=record(V/'source/instrumented_reference_executor_v2.py'),final_fixed_executor_sha256=record(F/'source/executor.py'),final_dynamic_executor_sha256=record(D/'runtime/executor.py'),mechanism_instrumented_executor_sha256=record(M/'runtime/executor.py'),install_dynamic_forward_AST_identical=True,changes='explicit checkpoint/config/load validation; configurable chunk with current protocol=1024; metadata fields; progress baseline from own records; removed legacy entrypoints. Functional forward unchanged.',validation_rows=valid,checkpoint_identity=vsum['authority']['checkpoint_identity'],checkpoint_weights_sha256=dp['model_sha256'],bounds='Original numerical tests are on original/instrumented V2 hashes, not a claim every adapter has rerun every test. Static common path means all5 functional routes with update-mask branch, not identical active/inactive operations.')
put('EXECUTOR_LINEAGE_AUDIT.json',lineage)
print('EXECUTOR_VALIDATION_EXTRACTED',[(r['test_name'],r['N'],r['pass_n'],r['status']) for r in valid],flush=True)
# Additional fields needed to replace the manuscript's subgroup/appendix tables without old-number carryover.
byco={c['sample_id']:c for c in cohort};r2groups=[]
for label,predicate in [('Native one switch',lambda c:c['switch_count']==1),('Native multiple switches',lambda c:c['switch_count']>1),('Multi-switch, full score',lambda c:c['switch_count']>1 and c['native_score']==1),('Multi-switch, below full score',lambda c:c['switch_count']>1 and c['native_score']<1)]:
 cc=[c for c in cohort if predicate(c)];r2groups.append(dict(cohort=label,N=len(cc),match_n=sum(r2[c['sample_id']]['matches_native'] for c in cc),Native_mean=mean([c['native_score'] for c in cc])*100,R2_mean=mean([r2[c['sample_id']]['r2_score'] for c in cc])*100))
tsv('TABLE_R2_SUBGROUPS_V2.tsv',r2groups)
nontriv=[c for c in cohort if c['nontrivial_reset']];k_non=mstat([(c['task'],core[c['sample_id']]['K']['K1']['score'],core[c['sample_id']]['K']['K0']['score']) for c in nontriv]);kclosure=ms['K_closure'];assert kclosure['K1_trace_exact']==165 and kclosure['K0_trace_exact']==110
additional=dict(deletion_branch_n=sum(len(x['deletion']) for x in core.values()),sham_n=sum(bool(x['sham']) for x in core.values()),all_OFF_prefix_n=sum(c['reset_eligible'] and not c['nontrivial_reset'] for c in cohort),K_closure=kclosure,K_nontrivial=k_non,r2_subgroups=r2groups,native_at_full_score_n=sum(c['native_score']==1 for c in cohort),native_below_full_score_n=sum(c['native_score']<1 for c in cohort),fixed13_aggregates={a:mean([fixed[s][a] for s in ids])*100 for a in ACT},fixed12_aggregates={a:mean([fixed[s][a] for s in ids if bt[s]!=CWE])*100 for a in ACT},dynamic_candidate_record_n=candidate_total,searched_improvement_pct=len(improved)/len(searched)*100,parent12_improvement_pct=len(improved)/len(r12)*100,parent13_improvement_pct=len(improved)/len(rows)*100)
additional['best_global_single_layer13']=max(ACT[1:-1],key=lambda a:additional['fixed13_aggregates'][a])
additional['best_global_single_layer12']=max(ACT[1:-1],key=lambda a:additional['fixed12_aggregates'][a])
put('SUPPLEMENTARY_WRITER_FIELDS.json',additional)
# Actual PDF authority: supplied by user; title, 20 pages and content are verified; filename year is not publication authority.
pdf=Path('/home/zonghan/.codex/attachments/d07adb75-9cd3-4d3c-b018-769fc74bc4ac/ICLR2026_Zonghan.pdf');pdfsha=record(pdf);pdfinfo=subprocess.check_output(['pdfinfo',str(pdf)],text=True);assert 'Pages:           20' in pdfinfo
pdftext=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);pages=pdftext.split('\f');assert 'T RAJECTORY AND S TATE' in pages[0] and '1.522' in pages[0] and '1,435' in pages[0];(O/'MANUSCRIPT_EXTRACTED.txt').write_text(pdftext)
manuscript=dict(status='FOUND_AND_VERIFIED',path=str(pdf),sha256=pdfsha,pages=20,title='BEYOND STATIC WRITES: TRAJECTORY AND STATE DIAGNOSTICS FOR IN-PLACE TEST-TIME TRAINING',header='Under review as a conference paper at ICLR 2027',note='Filename ICLR2026; actual document header ICLR2027. No newer manuscript inferred. Old numbers only supply replacement locations, never numerical statistical inputs.')
replacements=[]
def repl(page,label,old,field,new,status='REPLACE'):
 assert 1<=page<=20
 # Preserve the exact supplied old fragment when present, otherwise attach the printed table/paragraph location.
 replacements.append(dict(old_location=f'PDF p.{page}: {label}',old_label=label,old_value=old,new_authority_field=field,new_value=json.dumps(new,ensure_ascii=False),status=status))
for page,label in [(1,'Abstract'),(1,'Introduction'),(3,'Table1 full12 / Section3'),(4,'Figure1 full12'),(13,'Appendix B.3 bootstrap'),(14,'Appendix C full12')]:
 repl(page,label,'N=6000; SB84.44;Dynamic85.96;gain1.522;CI[1.270,1.788]','main_v2_dynamic',main)
for page,label in [(1,'Abstract searched'),(3,'Table1 searched / Section2.3'),(4,'Figure1 searched'),(13,'Appendix B.3 searched'),(15,'Table9 searched')]:
 repl(page,label,'1435;34.93;41.30;6.362;CI[5.331,7.433];173 improved;1262 equal','main_v2_dynamic.searched',main['searched'])
repl(3,'Table1 improved-only','173;40.48;93.26;+52.775','main_v2_dynamic.improved_only',main['improved_only'])
repl(1,'Abstract scope transition','Across 6000 examples, Dynamic gain1.522','full13_caseB',full)
repl(3,'Section2.3 denominator and cohorts','4565 ceiling;1435 searched;173 improved;167 early steady;151 nontrivial','main_v2_dynamic.ceiling_n;mechanism_v2',dict(ceiling=main['ceiling_n'],searched=len(searched),improved=len(improved),reset=165,nontrivial=157))
repl(4,'Section3 improvement prevalence','12.06%;2.88%;52.77 pp','supplementary',dict(searched_pct=additional['searched_improvement_pct'],population12_pct=additional['parent12_improvement_pct'],population13_pct=additional['parent13_improvement_pct'],improved_gain=main['improved_only']['gain_pp']))
repl(3,'Section3 full13 fixed means','OFF77.22;L24 77.32;ALL77.20;SB78.35','supplementary.fixed13_aggregates',dict(additional['fixed13_aggregates'],SampleBest=full['sample_best']))
repl(3,'Section3 common12 fixed means','OFF83.24;L24 83.35;ALL83.21;SB84.44','supplementary.fixed12_aggregates',dict(additional['fixed12_aggregates'],SampleBest=main['sample_best']))
repl(3,'Section3 best globally fixed single layer','L24','supplementary.best_global_single_layer13',dict(action=additional['best_global_single_layer13'],score13=additional['fixed13_aggregates'][additional['best_global_single_layer13']],score12=additional['fixed12_aggregates'][additional['best_global_single_layer12']]))
repl(3,'Section2.3 CWE','OFF4.94;SB5.24','cwe',cwe,'KEEP_IF_VERIFIED')
repl(14,'Appendix C Case B full6500','SB78.35;gain1.4046154','full13_caseB',full)
repl(14,'Appendix C fallback formula','6000/6500 * 1.5216667 = 1.4046154','full13_caseB.gain_pp',full['gain_pp'])
for row,old in zip(mechtable[:4],['173 40.48 93.26 41.71 51.55 [46.53,56.63] 163/10/0','167 41.22 93.01 49.15 43.86 [38.54,49.36] 135/32/0','151 40.47 92.94 44.43 48.51 [43.12,54.07] 135/16/0','151 40.47 92.94 51.77 41.17 [36.52,45.97] 134/17/0']):
 repl(5,'Table2 '+row['experiment'],old,'mechanism_v2.'+row['experiment'],row)
repl(4,'Section4.1 Reverse narrative','173;93.26 -> 41.71;163 decreases/10 ties','mechanism_v2.reverse',mechtable[0])
repl(5,'Section4.2 nontrivial prefix','16 all-OFF prefixes;151 nontrivial','supplementary.all_OFF_prefix_n',dict(all_OFF_prefix_n=additional['all_OFF_prefix_n'],nontrivial_n=157))
for p in [5,16]:repl(p,'Deletion branches / Sham','151 samples;359 branches;99 shams','supplementary',dict(deletion_n=157,branch_n=additional['deletion_branch_n'],sham_n=additional['sham_n']))
for p,label in [(5,'Section5.1 four-cell factorial Eq4'),(6,'Table3 / Section5 P path effect'),(17,'Appendix E / Table11 old L1-L6 closure'),(18,'Table12 F11-F10 path effect'),(18,'Table13 nontrivial K/P')]:
 repl(p,label,'P0/P1;F00/F01/F10/F11;167 or151 historical cohorts','claim_boundaries.P_omitted','Delete historical P factorial and its numeric effects from primary V2 mechanism; replace K-only descriptions and recorded closure definitions.','DELETE')
repl(6,'Table3 K-only replacement','F01 77.13;F11 93.01;closure108/167 and167/167','mechanism_v2.K0/K1; supplementary.K_closure',dict(K0=mechtable[5],K1=mechtable[6],closure=kclosure))
repl(6,'Section5.2 cache contrast','F11-F01=15.89;CI[11.66,20.49];52/115/0','mechanism_v2.K0',mechtable[5])
repl(18,'Table12 nontrivial cache contrast','151;17.57;CI[12.96,22.45];52/99/0','supplementary.K_nontrivial',k_non)
repl(17,'Table11 historical closure tiers','L1-L6;167 old cases','executor_validation.K_closure_scope','Replace with current explicitly recorded first-continuation attention/MLP/residual, final suffix hidden/logits/prediction; do not relabel as complete historical L1-L6.')
for p in [6,18]:repl(p,'R2 total / means','173;99120;SB40.48;Native93.26;R2 88.76;match160;unmatched13','mechanism_v2.r2;r2_structure',dict(table=mechtable[-1],counts=ms['R2']))
for p in [1,6,7,18]:repl(p,'R2 native multiple switches','60/73;82.19%;unmatched13','r2_structure',dict(match=62,multi=78,match_pct=62/78*100,unmatched=16))
oldsub=['100 100 93.20 93.20','73 60 93.33 82.67','52 44 100.00 88.11','21 16 76.83 69.21']
for row,old in zip(r2groups,oldsub):repl(7,'Table4 '+row['cohort'],old,'supplementary.r2_subgroups',row)
repl(18,'Appendix F old OFF-referenced recovery','91.66%; reference OFF','r2_structure.R2_RATIO_OF_SUMS_RECOVERY',dict(value=ratio,reference='Sample Best; redefine equation, do not just replace number'))
repl(13,'Appendix B.3 bootstrap protocol','seed20260913; historical cohort ordering;unstratified mechanism bootstrap','bootstrap',dict(dynamic_and_caseB=bootstrap,mechanism='20000 paired task-stratified, seed20260921, COHORT order; summarize.py',r2_structure='seed20260923;4000 mean draws;2000 correlation draws'))
repl(19,'Appendix G unresolved execution validation','These are validation requirements, not completed results','executor_validation',dict(table='TABLE_V2_EXECUTOR_VALIDATION.tsv',original_precheck=pre,scorer_isolation='NOT_FOUND',final_adapter_full_precheck='NOT_FOUND; static lineage and bridges available'))
repl(19,'Appendix G trajectory-dependent materialization','future actions route unused candidates via standard MLP','executor_validation.final_dynamic_executor_sha256',dict(sha=dp['executor_sha256'],write_only='STATIC_VERIFIED',all5_common_path='STATIC_VERIFIED',numerical_validation='original V2 precheck + final bridges'))
repl(4,'Section3 missing candidate-count claim','actual candidate-count and latency records ... absent','supplementary.dynamic_candidate_record_n',dict(candidate_records=candidate_total,random_baseline='ONGOING_EXCLUDED',raw_record_latency='AVAILABLE_IN_CURRENT_CANDIDATES'))
repl(20,'Appendix H reproducibility package','old173/167/151;1435 winning sequences;old analysis/recompute.py','provenance',dict(writer_package=str(O),searched=len(searched),improved=len(improved),reset=165,nontrivial=157))
# Existing fixed and task tables: preserve actual old PDF row text, point to complete replacement row.
aliases={'ruler_cwe_16k':'CWE','ruler_fwe_16k':'FWE','ruler_niah_multikey_1_16k':'NIAH Multikey 1','ruler_niah_multikey_2_16k':'NIAH Multikey 2','ruler_niah_multikey_3_16k':'NIAH Multikey 3','ruler_niah_multiquery_16k':'NIAH Multiquery','ruler_niah_multivalue_16k':'NIAH Multivalue','ruler_niah_single_1_16k':'NIAH Single 1','ruler_niah_single_2_16k':'NIAH Single 2','ruler_niah_single_3_16k':'NIAH Single 3','ruler_qa_hotpotqa_16k':'HotpotQA','ruler_qa_squad_16k':'SQuAD','ruler_vt_16k':'VT'}
for row in fixedtable:
 label=aliases.get(row['task'],row['task']);matches=[l.strip() for l in pages[13].splitlines() if label.lower() in l.lower()];repl(14,'Table7 '+row['task'],matches[0] if matches else 'see full Table7 row','fixed13.'+row['task'],row)
for row in tasktable:
 label=aliases.get(row['task'],row['task']);matches=[l.strip() for l in pages[13].splitlines() if label.lower() in l.lower()];repl(14,'Table8 '+row['task'],matches[-1] if len(matches)>1 else ('NEW CWE fallback row' if row['task']==CWE else 'see Table8'),'task_results.'+row['task'],row)
repl(15,'Figure2 task conditional gains','Old task-level searched gains','task_results',tasktable)
repl(15,'Table9 cohort tree','6500 fixed;6000 dynamic;1435 searched;173 improved;167 early steady;151 nontrivial','full13_caseB/main_v2_dynamic/mechanism_v2',dict(fixed=6500,composite=6500,dynamic12=6000,searched=1433,improved=170,reset=165,nontrivial=157))
tsv('MANUSCRIPT_NUMBER_REPLACEMENT_MAP.tsv',replacements);put('MANUSCRIPT_AUTHORITY.json',manuscript)
# Missing empirical evidence is an explicit writing limitation; extraction itself is complete.
missing=['V2 formal scorer-isolation intervention result: NOT_FOUND. Appendix G must retain this specific unverified item.']
limits=['Final adapted executor hash has no complete rerun of every original precheck. Report original exact prechecks, unchanged functional-forward AST and final bridges separately; do not claim a full identical-hash validation suite.','Current K trace closure does not provide historical complete L1-L6 tiers; remove those old tiers.','Historical validation used 173/1435 cohorts. These counts label validation only, never latest main/mechanism result cohorts.']
readiness=dict(MAIN_TABLE_READY='YES',CASE_B_READY='YES',MECHANISM_TABLE_READY='YES',R2_SECTION_READY='YES',EXECUTION_APPENDIX_READY='NO',EXECUTION_APPENDIX_WITH_EXPLICIT_LIMITATIONS_READY='YES')
# All input hashes are rechecked at finish, including actual raw fixed and Dynamic/mechanism files.
checks=dict(FIXED13_SAMPLE_N=len(fixed),FIXED13_CELL_N=len(cellmeta),FIXED13_MISSING_N=0,FIXED13_DUPLICATE_N=0,FIXED13_SCORE_PARSE_ERROR_N=0,FIXED13_SAMPLE_COLLISION_N=0,fixed_underlying_new_cells=raw_new,fixed_underlying_reused_cells=len(reusekeys),DYN12_CI_REPRODUCED_EXACT=True,SEARCHED_CI_REPRODUCED_EXACT=True,MECHANISM_MEANS_COUNTS_CIS_REPRODUCED_EXACT=True,DYNAMIC_SOURCE_FILES=len(records),DYNAMIC_CANDIDATE_RECORDS=candidate_total,R2_CANDIDATES=totalcand,R2_REUSED_PARITY=reuse_n,source_authority_conflict=False,GPU_USED='NO',MODEL_LOADED='NO',OLD_V1_USED_AS_STATISTICAL_INPUT='NO',FORMAL_RESULTS_MODIFIED='NO',CPU_AFFINITY=sorted(os.sched_getaffinity(0)),NUMPY=np.__version__)
# Include every relevant metadata/source consulted during discovery as well as statistical inputs.
for root in [E,E/'completion']:
 for p in root.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.py','.md']:record(p)
record(Path('/home/zonghan/.codex/attachments/8a0aa3bf-3d46-40a6-bef6-dfff706b6e68/已粘贴的文本.txt'))
for p in [B/'audits/r2_mechanism_cpu_20260923/NOT_RUN.md',V/'V2_SOURCE_DIFF.patch']:record(p)
for p in [F/'reused/FROZEN_V2_STAGE2_PROTOCOL.json',D/'README.md',M/'REPORT.md']:
 if p.exists():record(p)
upstream_n=0
for root in [E,E/'completion']:
 upstream=j(root/'input_provenance.json')
 if isinstance(upstream,dict):
  for path,meta in upstream.items():
   if isinstance(meta,dict) and 'sha256' in meta and path.startswith('/'):
    assert record(Path(path))==meta['sha256'],path
    upstream_n+=1
checks['UPSTREAM_CPU_INPUT_HASHES_MATCHED']=upstream_n
print('REHASHING_INPUTS',len(prov),flush=True)
changed=[]
for p,q in prov.items():
 s=Path(p).stat()
 if s.st_size!=q['size'] or s.st_mtime_ns!=q['mtime_ns'] or digest(p)!=q['sha256']:changed.append(p)
assert not changed,changed
checks['INPUT_FILES_VERIFIED_UNCHANGED']=len(prov);checks['INPUT_FILES_CHANGED']=changed
put('INPUT_PROVENANCE.json',dict(files=list(prov.values()),input_file_n=len(prov),all_unchanged=True,weights_file_not_loaded=True,weight_identity_source='frozen protocols + raw records + completed validation hash manifests; no model deserialization',old_V1_results='not read as statistical inputs; original validation reports may reference V1 solely as historical provenance'))
put('AUDIT_CHECKS.json',checks)
handoff=dict(main_v2_dynamic=main,full13_caseB=full,cwe=cwe,task_results=tasktable,fixed13=fixedtable,mechanism_v2={r['experiment']:r for r in mechtable},r2_structure=rstats,executor_validation=lineage,claim_boundaries=boundaries,supplementary=additional,bootstrap=bootstrap,manuscript=manuscript,readiness=readiness,BLOCKING_MISSING=missing,nonblocking_limitations=limits,provenance=dict(authority_map=str(O/'WRITER_AUTHORITY_MAP.md'),input_manifest=str(O/'INPUT_PROVENANCE.json'),checks=checks),STATUS='PARTIAL',EXTRACTION_COMPLETE=True)
put('WRITER_DATA_HANDOFF.json',handoff)
lines=['# READY-TO-USE NUMBERS','', '本包为已完成正式 V2 artifact 的独立 CPU 写作转接包。数值提取完成；完整 execution appendix 验证声明仍为 PARTIAL，具体缺口在最后一节。','','| Scope | N | Sample Best | Sequential / Dynamic | Gain pp | 95% CI pp |','|---|---:|---:|---:|---:|---|']
for label,s in [('Full13 Case B',full),('Measured-task parent12',main),('Searched non-ceiling',main['searched'])]:lines.append(f"| {label} | {s['N']} | {s['sample_best']:.8f} | {s['sequential_score']:.8f} | {s['gain_pp']:.8f} | {s['gain_ci95_pp']} |")
lines+=['',f"CWE: N={cwe['N']}, OFF={cwe['OFF']:.8f}, SampleBest={cwe['SampleBest']:.8f}. CWE_DYNAMIC_MEASURED=NO; CWE_INCREMENTAL_CREDIT=ZERO_BY_FALLBACK. 这不是测得的零 Dynamic 增益。",'',f"12-task ceiling={main['ceiling_n']}; searched={len(searched)}; improved={len(improved)}; equal={main['searched']['equal_n']}; worse=0. Improved-only SB={main['improved_only']['sample_best']:.8f}, Native={main['improved_only']['sequential_score']:.8f}, gain={main['improved_only']['gain_pp']:.8f} pp; OUTCOME_CONDITIONED=YES.",'',f"Best global single-layer action is now {additional['best_global_single_layer13']} (13-task {additional['fixed13_aggregates'][additional['best_global_single_layer13']]:.8f}); replace the manuscript's old L24 label as well as its numeric score.",'', 'Full13 CI directly resamples all 13 task strata, retaining CWE zero-delta draws. It is not obtained by scaling a rounded 12-task interval. Sample-level tables use [0,1]; aggregate means [0,100]; differences pp. Full13 is a protocol-defined composite; non-CWE ceiling rows are also carried, not newly inferred Dynamic measurements.','','| Mechanism | N | SB | Native | Condition | Drop pp | 95% CI | ↓ / = / ↑ |','|---|---:|---:|---:|---:|---:|---|---|']
for r in mechtable:lines.append(f"| {r['experiment']} | {r['N']} | {r['SampleBest_mean_if_available']:.4f} | {r['Native_mean']:.4f} | {r['Condition_mean']:.4f} | {r['paired_effect_pp']:.4f} | [{r['CI_low']:.4f}, {r['CI_high']:.4f}] | {r['decrease_n']}/{r['equal_n']}/{r['increase_n']} |")
lines+=['',f"R2: {totalcand} exhaustive candidates; one-switch Native92, multi78; matches154/170 and62/78. Mean per-sample recovery={rstats['R2_MEAN_PER_SAMPLE_RECOVERY']:.12f}, CI={rstats['R2_MEAN_RECOVERY_CI']}; ratio-of-sums={ratio:.12f}. Matched-multi multiple solutions50, unique12, median4; adjacent boundary22/62, isolated40/62. Prediction hash any-match152/154 and60/62.",'',f"Set-wise reversal: all matched {rstats['REVERSAL_ALL_MATCHED_MEAN_PP']:.8f} pp {rstats['REVERSAL_ALL_MATCHED_CI']}; matched-multi {rstats['REVERSAL_MULTI_MATCHED_MEAN_PP']:.8f} pp {rstats['REVERSAL_MULTI_MATCHED_CI']}; all matches beat reverse54/62. Global {totalpairs} unordered pairs: sample-equal score-different {rstats['GLOBAL_PAIR_SCORE_DIFFERENT_PCT']:.8f}%, mean absolute difference {rstats['GLOBAL_ABS_DIFF_MEAN_PP']:.8f} pp.",'','Original V2 exact prechecks: same-path4/4, zero-write64/64, future-prefix15/15, apply-then-update5/5; precheck checkpoint/state144/144. Stage-1 checkpoint/state1384/1384, Stage-2 checkpoint/state11480/11480. Counts are comparisons or trajectories as specified, not new independent samples. Final adapter uses the AST-identical functional forward; numerical prechecks are not claimed to have rerun under its different hash.','', '# MANUSCRIPT REPLACEMENTS','',f"Actual supplied PDF: {pdf}; SHA256 {pdfsha}; 20 pages. Filename says ICLR2026 but rendered header says ICLR2027. It is used only to locate replacements.",'',f"See MANUSCRIPT_NUMBER_REPLACEMENT_MAP.tsv ({len(replacements)} entries): update Abstract/Table1/Figure1, replace full Table7/8, replace Table2 mechanism and Table4 R2 subgroups, delete P/four-cell factorial effects and old closure tiers; replace Appendix C with full6500 Case B and new CI; rewrite Appendix B.3 bootstrap; partially resolve Appendix G with explicit remaining scorer-test gap.",'', '# CLAIM BOUNDARIES','']
lines += ['- NOT_SUPPORTED: '+s for s in boundaries['NOT_SUPPORTED']]
lines += ['', 'All mechanism and R2 structural conclusions are conditional on current Dynamic improvement. Score-selected reversal differences are not unbiased causal effects; prediction hashes do not establish state equivalence. K closure only covers recorded sites. CPU R2 analysis CI settings differ from main-effect settings and are recorded separately. Random baseline is ongoing and not incorporated.', '', '# STILL MISSING','', 'BLOCKING_MISSING: '+missing[0],'']+limits+['','MAIN_TABLE_READY=YES; CASE_B_READY=YES; MECHANISM_TABLE_READY=YES; R2_SECTION_READY=YES; EXECUTION_APPENDIX_READY=NO. An appendix accurately reporting validation boundaries is ready; an all-gates-PASS assertion is not.','',f"Integrity: GPU_USED=NO; MODEL_LOADED=NO; FORMAL_RESULTS_MODIFIED=NO; OLD_V1_USED_AS_STATISTICAL_INPUT=NO. {len(prov)} source files rehashed unchanged. CPU affinity 8–23. Fullprecision numbers in WRITER_DATA_HANDOFF.json; requested TSVs, input hashes and SHA256SUMS included."]
(O/'WRITER_DATA_HANDOFF.md').write_text('\n'.join(lines)+'\n')
summary=['FINAL_WRITER_DATA_EXTRACTION_COMPLETE','','[V2_DYNAMIC]',f'DYN12_PARENT_N={main["N"]}',f'DYN12_TASK_N={main["task_n"]}',f'DYN12_SB_MEAN={main["sample_best"]}',f'DYN12_DYNAMIC_MEAN={main["sequential_score"]}',f'DYN12_GAIN_PP={main["gain_pp"]}',f'DYN12_GAIN_CI={main["gain_ci95_pp"]}',f'DYN12_CEILING_N={main["ceiling_n"]}',f'DYN12_SEARCHED_N={len(searched)}',f'SEARCHED_SB_MEAN={main["searched"]["sample_best"]}',f'SEARCHED_DYNAMIC_MEAN={main["searched"]["sequential_score"]}',f'SEARCHED_GAIN_PP={main["searched"]["gain_pp"]}',f'SEARCHED_GAIN_CI={main["searched"]["gain_ci95_pp"]}',f'DYN12_IMPROVED_N={len(improved)}',f'DYN12_EQUAL_N={main["searched"]["equal_n"]}','DYN12_WORSE_N=0','','[FULL13_CASE_B]',f'FIXED13_SAMPLE_N={len(fixed)}',f'FIXED13_CELL_N={len(cellmeta)}','FIXED13_MISSING_N=0','FIXED13_DUPLICATE_N=0',f'FULL13_SB_MEAN={full["sample_best"]}',f'FULL13_SEQUENTIAL_LB_MEAN={full["sequential_score"]}',f'FULL13_GAIN_PP={full["gain_pp"]}',f'FULL13_GAIN_CI={full["gain_ci95_pp"]}',f'CWE_SB_MEAN={cwe["SampleBest"]}','CWE_DYNAMIC_MEASURED=NO','CWE_INCREMENTAL_CREDIT=ZERO_BY_FALLBACK','','[MECHANISM]','REVERSE_N=170','RESET_N=165','NONTRIVIAL_RESET_N=157','DELETION_N=157','K_N=165','R2_N=170','','[R2]','MULTI_SWITCH_N=78','MULTI_R2_MATCH_N=62','MULTI_R2_UNMATCHED_N=16',f'TOTAL_GAIN_RECOVERY_RATIO={ratio}','ADJACENT_BOUNDARY_N=22','ISOLATED_BOUNDARY_N=40','MULTI_PREDICTION_MATCH_N=60','','[EXECUTOR]',f'V2_SOURCE_SHA={dp["executor_sha256"]}',f'ORIGINAL_VALIDATED_V2_SOURCE_SHA={vsum["source_sha256"]}','ACTION_CONTROLS_WRITE_ONLY=STATIC_VERIFIED','COMMON_PATH_VALIDATED=ORIGINAL_PRECHECK_EXACT_PLUS_FINAL_STATIC_LINEAGE_AND_BRIDGES','ZERO_WRITE_VALIDATION=ORIGINAL_V2_64/64','PREFIX_INVARIANCE=ORIGINAL_V2_15/15','APPLY_THEN_UPDATE=ORIGINAL_V2_5/5','SCORER_ISOLATION=NOT_FOUND','','[WRITER]']+[k+'='+v for k,v in readiness.items()]+['','[INTEGRITY]','GPU_USED=NO','MODEL_LOADED=NO','OLD_V1_STATISTICAL_INPUT=NO','FORMAL_RESULTS_MODIFIED=NO',f'INPUT_FILES_REHASHED_UNCHANGED={len(prov)}','','[FINAL]','STATUS=PARTIAL','BLOCKING_MISSING='+missing[0]]
(O/'FINAL_REQUIRED_SUMMARY.txt').write_text('\n'.join(summary)+'\n')
# Manifest excludes itself by definition. No output will be changed after hashing.
(O/'SHA256SUMS').write_text('\n'.join(digest(p)+'  '+str(p.relative_to(O)) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')+'\n')
print('\n'.join(summary),flush=True)
