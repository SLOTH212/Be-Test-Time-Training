#!/usr/bin/env python3
"""Generic Dynamic LB aggregation with ceiling reconstruction and paired bootstrap."""

import argparse,json,statistics,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from dynamic_ttt.eval.runtime import TOL,atomic_json,dynamic_required,fixed_actions,load_config


def aggregate(fixed_aggregate,dynamic_dir,layers,excluded_tasks,draws=10000,seed=20260828):
    actions=fixed_actions(layers);rows=[r for r in fixed_aggregate['per_sample'] if r['task'] not in excluded_tasks];dynamic={}
    for path in Path(dynamic_dir).glob('*.json'):
        rec=json.loads(path.read_text());sid=rec['sample_id']
        if sid in dynamic:raise ValueError('duplicate Dynamic result')
        dynamic[sid]=rec
    table=[]
    for row in rows:
        sb=float(row['sample_best']);required=dynamic_required(sb)
        if required:
            if row['sample_id'] not in dynamic or dynamic[row['sample_id']].get('search_status')!='SUCCESS':raise ValueError('missing/invalid Dynamic result')
            score=float(dynamic[row['sample_id']]['dynamic_score'])
        else:score=sb
        table.append({"sample_id":row['sample_id'],"task":row['task'],"OFF":row['OFF'],"SampleBest":sb,"Dynamic":score,"delta":score-sb,"searched":required})
    if set(dynamic)!={r['sample_id'] for r in table if r['searched']}:raise ValueError('extra Dynamic result')
    values=np.asarray([r['delta'] for r in table]);rng=np.random.default_rng(seed);boot=np.zeros(draws)
    for task in sorted({r['task'] for r in table}):
        x=np.asarray([r['delta'] for r in table if r['task']==task]);idx=rng.integers(0,len(x),size=(draws,len(x)));boot+=x[idx].sum(axis=1)
    boot/=len(table)
    return {"sample_count":len(table),"searched_count":sum(r['searched'] for r in table),"ceiling_count":sum(not r['searched'] for r in table),"off_mean":statistics.fmean(r['OFF'] for r in table),"sample_best_mean":statistics.fmean(r['SampleBest'] for r in table),"dynamic_mean":statistics.fmean(r['Dynamic'] for r in table),"dynamic_minus_sample_best":statistics.fmean(r['delta'] for r in table),"improved":sum(r['delta']>TOL for r in table if r['searched']),"equal":sum(abs(r['delta'])<=TOL for r in table if r['searched']),"worse":sum(r['delta']<-TOL for r in table if r['searched']),"bootstrap":{"draws":draws,"seed":seed,"method":"task_stratified_paired","ci95_lower":float(np.quantile(boot,.025,method='linear')),"ci95_upper":float(np.quantile(boot,.975,method='linear')),"probability_delta_gt_zero":float(np.mean(boot>0))},"per_sample":table}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--fixed-aggregate',required=True);p.add_argument('--dynamic-results',required=True);p.add_argument('--output',required=True);a=p.parse_args();cfg=load_config(a.config);fixed=json.loads(Path(a.fixed_aggregate).read_text());out=aggregate(fixed,a.dynamic_results,cfg['ttt_layers'],set(cfg.get('dynamic_excluded_tasks',['ruler_cwe_16k'])));atomic_json(a.output,out);print(json.dumps({k:v for k,v in out.items() if k!='per_sample'},sort_keys=True))


if __name__=='__main__':main()
