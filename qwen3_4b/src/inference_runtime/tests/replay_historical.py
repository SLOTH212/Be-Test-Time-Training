#!/usr/bin/env python3
"""No-GPU replay of completed 1.7B Fixed7 and Dynamic result authorities."""
import argparse,csv,importlib.util,json,math,statistics,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def close(a,b,tol=1e-12):return abs(float(a)-float(b))<=tol

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixed-run',required=True);p.add_argument('--dynamic-run',required=True);p.add_argument('--output');a=p.parse_args();fixed=Path(a.fixed_run);dynamic=Path(a.dynamic_run)
    agg=module(ROOT/'fixed/aggregate_fixed.py','generic_fixed_aggregate');rows=[]
    for action in ['OFF','L0','L6','L12','L18','L24','ALL']:
        with (fixed/'modes'/action/'results.jsonl').open(encoding='utf-8') as stream:rows.extend(json.loads(line) for line in stream if line.strip())
    reconstructed=agg.aggregate(rows,[0,6,12,18,24]);authority=json.loads((fixed/'aggregation/fixed7_summary.json').read_text())
    fixed_checks={"OFF":close(reconstructed['action_means']['OFF'],authority['OFF_SCORE']),"BEST_SINGLE_ACTION":reconstructed['best_single_action']==authority['BEST_SINGLE_LAYER'],"BEST_SINGLE":close(reconstructed['best_single_mean'],authority['BEST_SINGLE_SCORE']),"SAMPLE_BEST":close(reconstructed['sample_best_mean'],authority['SAMPLE_BEST_SCORE']),"COUNT":reconstructed['sample_count']==6500}
    with (dynamic/'eligibility/fixed7_sample_best_full6500.csv').open(newline='',encoding='utf-8') as stream:all_rows=list(csv.DictReader(stream))
    rows12=[r for r in all_rows if r['task']!='ruler_cwe_16k'];result_by_id={}
    for path in (dynamic/'results').glob('*.json'):
        r=json.loads(path.read_text());result_by_id[r['sample_id']]=r
    rebuilt=[]
    for r in rows12:
        sb=float(r['sample_best']);executed=r['dynamic_eligible']=='true';score=float(result_by_id[r['sample_id']]['dynamic_score']) if executed else sb
        rebuilt.append({"task":r['task'],"off":float(r['OFF']),"sb":sb,"dynamic":score,"delta":score-sb,"executed":executed})
    headline=json.loads((dynamic/'aggregation/dynamic_12task_headline.json').read_text());dyn=statistics.fmean(r['dynamic'] for r in rebuilt);sb=statistics.fmean(r['sb'] for r in rebuilt);off=statistics.fmean(r['off'] for r in rebuilt)
    rng=np.random.default_rng(20260828);draws=np.zeros(10000)
    for task in list(dict.fromkeys(r['task'] for r in rebuilt)):
        x=np.asarray([r['delta'] for r in rebuilt if r['task']==task]);idx=rng.integers(0,len(x),size=(10000,len(x)));draws+=x[idx].sum(axis=1)
    draws/=6000
    boot=json.loads((dynamic/'bootstrap/dynamic_vs_samplebest_task_stratified_10000.json').read_text())
    dynamic_checks={"COUNT":len(rebuilt)==6000 and len(result_by_id)==1435,"OFF":close(off,headline['OFF_12TASK_SCORE']),"SAMPLE_BEST":close(sb,headline['SAMPLE_BEST_12TASK_SCORE']),"DYNAMIC":close(dyn,headline['DYNAMIC_12TASK_SCORE']),"DELTA":close(100*(dyn-sb),headline['DYNAMIC_MINUS_SAMPLEBEST_12TASK_PP']),"BOOTSTRAP_MEAN":close(draws.mean(),boot['bootstrap_mean_delta']),"BOOTSTRAP_LOW":close(np.quantile(draws,.025,method='linear'),boot['ci95_lower']),"BOOTSTRAP_HIGH":close(np.quantile(draws,.975,method='linear'),boot['ci95_upper'])}
    out={"status":"PASS" if all(fixed_checks.values()) and all(dynamic_checks.values()) else "FAIL","HISTORICAL_FIXED_AGGREGATION_PARITY":"PASS" if all(fixed_checks.values()) else "FAIL","HISTORICAL_DYNAMIC_AGGREGATION_PARITY":"PASS" if all(dynamic_checks.values()) else "FAIL","FORMAL_RESULT_REPLAY_STATUS":"PASS" if all(fixed_checks.values()) and all(dynamic_checks.values()) else "FAIL","fixed_checks":fixed_checks,"dynamic_checks":dynamic_checks,"reconstructed":{"fixed_sample_best":reconstructed['sample_best_mean'],"dynamic_mean":dyn,"dynamic_minus_sample_best_pp":100*(dyn-sb),"bootstrap_ci":[float(np.quantile(draws,.025,method='linear')),float(np.quantile(draws,.975,method='linear'))]}}
    if a.output:
        Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,sort_keys=True))
    if out['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
