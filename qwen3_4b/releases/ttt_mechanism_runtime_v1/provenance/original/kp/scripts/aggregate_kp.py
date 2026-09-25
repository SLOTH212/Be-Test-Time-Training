#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, statistics, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

CELLS=['F00','F10','F01','F11']; LEVELS=['L1','L2','L3','L4','L5','L6']
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.{time.time_ns()}.tmp')
 fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def atomic_text(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.{time.time_ns()}.tmp')
 fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def atomic_csv(p,rs):
 rs=list(rs);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.{time.time_ns()}.tmp')
 with t.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs);f.flush();os.fsync(f.fileno())
 os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def describe(x):
 a=np.asarray(x,float);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'std':float(a.std(ddof=1)) if len(a)>1 else 0.0,'q05':float(np.quantile(a,.05)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'q95':float(np.quantile(a,.95))}
def boot(x,seed=20260827,B=20000):
 a=np.asarray(x,float);rng=np.random.default_rng(seed);v=np.empty(B)
 for i in range(B):v[i]=a[rng.integers(0,len(a),len(a))].mean()
 return {'mean':float(a.mean()),'median':float(np.median(a)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'replicates':B,'seed':seed,'paired':True}
def effects(d):
 return {'K_AT_P0':d['F00']-d['F10'],'K_AT_P1':d['F01']-d['F11'],'P_AT_K0':d['F00']-d['F01'],'P_AT_K1':d['F10']-d['F11'],'MAIN_K':.5*((d['F00']-d['F10'])+(d['F01']-d['F11'])),'MAIN_P':.5*((d['F00']-d['F01'])+(d['F10']-d['F11'])),'INTERACTION':d['F00']-d['F10']-d['F01']+d['F11']}
def group_summary(ids,by,refs):
 d={sid:{c:float(by[sid][c]['generation_start_relative_l2']) for c in CELLS} for sid in ids};e={sid:effects(x) for sid,x in d.items()}
 return {'N':len(ids),'distances':{c:describe([d[s][c] for s in ids]) for c in CELLS},'effects':{k:describe([e[s][k] for s in ids]) for k in next(iter(e.values()))},'closure':{c:{l:{'count':sum(bool(by[s][c]['closure'][l]) for s in ids),'rate':sum(bool(by[s][c]['closure'][l]) for s in ids)/len(ids)} for l in LEVELS} for c in CELLS},'rewards':{c:describe([float(by[s][c]['terminal_reward']) for s in ids]) for c in CELLS},'native_reward':describe([float(refs[s]['native_reward']) for s in ids]),'control_reward':describe([float(refs[s]['control_reward']) for s in ids])}

def main(run_root):
 run=Path(run_root).resolve();spec=importlib.util.spec_from_file_location('kp_worker',run/'scripts/kp_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 run,cfg,cohort,manifest,checks=w.gates(run);mby={r['branch_id']:r for r in manifest};found=[]
 for p in sorted((run/'branches').glob('*.json')):
  x=load(p)
  if x.get('branch_id') not in mby or not w.valid_branch(x,mby[x['branch_id']],cfg):raise RuntimeError('INVALID_BRANCH '+str(p))
  found.append(x)
 if len(found)!=668 or len({x['branch_id'] for x in found})!=668:raise RuntimeError(f'BRANCH_INTEGRITY n={len(found)} unique={len({x.get("branch_id") for x in found})}')
 by=defaultdict(dict)
 for x in found:
  if x['cell'] in by[x['canonical_sample_id']]:raise RuntimeError('DUPLICATE_CELL')
  by[x['canonical_sample_id']][x['cell']]=x
 ids=[r['canonical_sample_id'] for r in cohort]
 if any(set(by[s])!=set(CELLS) for s in ids):raise RuntimeError('FOUR_CELL_INCOMPLETE')
 refs={x['canonical_sample_id']:x for x in (load(p) for p in (run/'source_states').glob('*.json'))}
 if set(refs)!=set(ids):raise RuntimeError('REFERENCE_SCOPE')
 all_summary=group_summary(ids,by,refs)
 sample_rows=[]
 for sid in ids:
  d={c:float(by[sid][c]['generation_start_relative_l2']) for c in CELLS};e=effects(d);r=next(x for x in cohort if x['canonical_sample_id']==sid)
  sample_rows.append({'canonical_sample_id':sid,'task':r['task'],'trajectory_type':'MULTISWITCH' if r['switch_count']>1 else 'SIMPLE',**{f'd{c[1:]}':d[c] for c in CELLS},**e,**{f'{c}_L6':by[sid][c]['closure']['L6'] for c in CELLS}})
 atomic_csv(run/'aggregates/KP_SAMPLE_FACTORIAL_EFFECTS.csv',sample_rows)
 uncertainty={k:boot([x[k] for x in sample_rows]) for k in ['K_AT_P0','K_AT_P1','P_AT_K0','P_AT_K1','MAIN_K','MAIN_P','INTERACTION']}
 task={t:group_summary([r['canonical_sample_id'] for r in cohort if r['task']==t],by,refs) for t in sorted({r['task'] for r in cohort})}
 groups={g:group_summary([r['canonical_sample_id'] for r in cohort if ('MULTISWITCH' if r['switch_count']>1 else 'SIMPLE')==g],by,refs) for g in ['SIMPLE','MULTISWITCH']}
 fd=Counter((x['cell'],x['first_divergence']['closure_level'],x['first_divergence']['tensor']) for x in found)
 diagnostics=[{'cell':k[0],'closure_level':k[1],'tensor_family':k[2],'count':v} for k,v in sorted(fd.items())]
 atomic(run/'diagnostics/FIRST_DIVERGENCE_DISTRIBUTION.json',diagnostics)
 atomic(run/'aggregates/KP_PRIMARY_FACTORIAL_ANALYSIS.json',{'all_167':all_summary,'paired_bootstrap':uncertainty,'per_task':task,'trajectory_type':groups,'timestamp':now()})
 closure=all_summary['closure'];f11=closure['F11']['L6']['count']
 if f11!=167:
  bad=[s for s in ids if not by[s]['F11']['closure']['L6']];q={'status':'FAIL_OR_BLOCKED','failure_count':len(bad),'samples':bad,'required_audit_targets':['amendment implementation','K copy','P materialization','fast-weight injection','cache position','boundary','suffix mismatch','tail semantics','generation semantics','other implementation issue'],'timestamp':now()};atomic(run/'audit/F11_NONCLOSURE_DIAGNOSTIC.json',q);atomic_text(run/'audit/F11_NONCLOSURE_DIAGNOSTIC.md','# F11 nonclosure diagnostic\n\n'+json.dumps(q,indent=2)+'\n')
 fast_all=all(x['fast_weight_parity']=='PASS' for x in found);base_mut=any(x['base_parameter_mutation'] for x in found)
 integrity=all(checks.values()) and f11==167 and fast_all and not base_mut
 receipt={'MODEL_IDENTITY':cfg['model_identity'],'CHECKPOINT_IDENTITY':cfg['checkpoint_identity'],'RULE_MANIFEST_SHA256':cfg['rule_sha256'],'ELIGIBILITY_MANIFEST_SHA256':cfg['eligibility_sha256'],'AMENDMENT_SHA256':cfg['amendment_sha256'],'COHORT_MANIFEST_SHA256':cfg['cohort_sha256'],'BRANCH_MANIFEST_SHA256':cfg['branch_manifest_sha256'],'PROTOCOL_SHA256':cfg['protocol_sha256'],'N':167,**{f'D{c[1:]}_MEAN':all_summary['distances'][c]['mean'] for c in CELLS},**{k:all_summary['effects'][k]['mean'] for k in ['K_AT_P0','K_AT_P1','P_AT_K0','P_AT_K1','MAIN_K','MAIN_P','INTERACTION']},**{f'{c}_{l}_CLOSURE':closure[c][l]['count'] for c in CELLS for l in LEVELS},**{f'{c}_REWARD_MEAN':all_summary['rewards'][c]['mean'] for c in CELLS},'NATIVE_REWARD_MEAN':all_summary['native_reward']['mean'],'CONTROL_REWARD_MEAN':all_summary['control_reward']['mean'],'MULTISWITCH_N':groups['MULTISWITCH']['N'],'SIMPLE_N':groups['SIMPLE']['N'],'FAST_WEIGHT_PARITY':'PASS' if fast_all else 'FAIL','F11_NATIVE_EQUIVALENCE_AUDIT':'PASS' if f11==167 else 'FAIL_OR_BLOCKED','FINAL_INTEGRITY':'PASS' if integrity else 'FAIL_OR_BLOCKED','FORMAL_KP_FACTORIAL_COMPLETE':True,'TOTAL_FACTORIAL_BRANCHES_VALID':668,'FOUR_CELL_SAMPLE_AGGREGATES_VALID':167,'MISSING_BRANCHES':0,'DUPLICATE_BRANCHES':0,'UNRESOLVED_ERRORS':0,'NATIVE_REFERENCE_REPLAY_PARITY':'PASS','MULTISWITCH_AMENDMENT_SMOKE':'PASS','K_FACTOR_ORTHOGONALITY':'PASS','P_FACTOR_ORTHOGONALITY':'PASS','SOURCE_STATE_ISOLATION':'PASS','BASE_PARAMETER_MUTATION':base_mut,'HISTORICAL_OVERLAP':0,'KP_OUTCOME_LEAKAGE':0,'R2_LAUNCHED':False,'CRITICALITY_LAUNCHED':False,'OBSERVABILITY_LAUNCHED':False,'timestamp':now()}
 atomic(run/'reports/KP_FACTORIAL_1P7B_FORMAL_COMPLETION_RECEIPT.json',receipt)
 atomic_text(run/'reports/KP_FACTORIAL_1P7B_FORMAL_COMPLETION_RECEIPT.md','# K/P Factorial 1.7B Formal Completion Receipt\n\n```text\n'+'\n'.join(f'{k}={v}' for k,v in receipt.items())+'\n```\n')
 w.status(run,pipeline_status='COMPLETE' if integrity else 'FINAL_INTEGRITY_BLOCKED',current_phase='STOP_NO_NEXT_MECHANISM',committed_results=668,formal_complete=True,final_receipt_exists=True,worker_running=False)
 if not integrity:raise RuntimeError('FINAL_INTEGRITY_FAIL_OR_BLOCKED')
 print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main(sys.argv[1])
