#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

MODES=['OFF','L0','L6','L12','L18','L24','ALL']; SINGLES=['L0','L6','L12','L18','L24']

def mean(xs): return sum(xs)/len(xs)

def aggregate(rows):
    by=defaultdict(dict); task_of={}
    for r in rows:
        sid,mode=r['sample_id'],r['mode']
        if mode not in MODES or mode in by[sid]: raise ValueError(f'duplicate/unknown mode {sid} {mode}')
        if r.get('status') not in ('ok','SUCCESS') or not isinstance(r.get('score'),(int,float)) or not math.isfinite(r['score']): raise ValueError(f'invalid row {sid} {mode}')
        if sid in task_of and task_of[sid]!=r['task']: raise ValueError(f'task mismatch {sid}')
        task_of[sid]=r['task']; by[sid][mode]=float(r['score'])
    if not by: raise ValueError('empty results')
    for sid,m in by.items():
        if set(m)!=set(MODES): raise ValueError(f'incomplete mode set for {sid}: {sorted(m)}')
    scores={m:mean([by[s][m] for s in sorted(by)]) for m in MODES}
    best=max(SINGLES,key=lambda m:(scores[m],-SINGLES.index(m)))
    sample_best={sid:max(vals.values()) for sid,vals in by.items()}; sample_best_score=mean(list(sample_best.values()))
    if sample_best_score+1e-15 < max(scores.values()): raise AssertionError('oracle monotonicity violated')
    tasks=sorted(set(task_of.values())); breakdown={}
    for task in tasks:
        ids=[sid for sid in sorted(by) if task_of[sid]==task]
        breakdown[task]={'OFF':mean([by[s]['OFF'] for s in ids]),'Best Single':mean([by[s][best] for s in ids]),
                         'ALL':mean([by[s]['ALL'] for s in ids]),'Sample Best':mean([sample_best[s] for s in ids]),'sample_count':len(ids)}
    return {'OFF_SCORE':scores['OFF'],**{f'{m}_SCORE':scores[m] for m in SINGLES},
            'BEST_SINGLE_LAYER':best,'BEST_SINGLE_SCORE':scores[best],'ALL_SCORE':scores['ALL'],
            'SAMPLE_BEST_SCORE':sample_best_score,'SAMPLE_COUNT':len(by),'TASK_COUNT':len(tasks),
            'TASK_BREAKDOWN':breakdown,'ALIGNMENT_KEY':'sample_id','BEST_SINGLE_SCOPE':'global_full_benchmark',
            'SAMPLE_BEST_SCOPE':'per_sample_max_over_OFF_L0_L6_L12_L18_L24_ALL','ORACLE_MONOTONICITY':'PASS'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in open(args.input,encoding='utf-8') if x.strip()]; out=aggregate(rows)
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise SystemExit(f'refusing to overwrite {p}')
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))

if __name__=='__main__': main()
