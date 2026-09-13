"""Recompute historical sequential means and task-stratified paired bootstrap."""
import argparse,csv,json,statistics
from collections import Counter
from pathlib import Path
from dynamic_ttt.stats.dynamic import formal_dynamic_bootstrap

def analyze(rows,seed=20260828,resamples=10000):
 if len(rows)!=6000 or len({r['sample_id'] for r in rows})!=6000:raise ValueError('FORMAL_POPULATION_REQUIRED')
 counts=Counter(r['task'] for r in rows)
 if len(counts)!=12 or set(counts.values())!={500}:raise ValueError('FORMAL_TASK_COUNTS_REQUIRED')
 methods=['OFF','BEST_SINGLE','ALL','SAMPLE_BEST','DYNAMIC']
 means={k:statistics.fmean(float(r[k]) for r in rows) for k in methods}
 pairs=[]
 for r in rows:
  sb=max(float(r[k]) for k in ['OFF','L0','L6','L12','L18','L24','ALL'])
  if abs(sb-float(r['SAMPLE_BEST']))>1e-12:raise ValueError('SAMPLE_BEST_PARITY')
  delta=float(r['DYNAMIC'])-sb
  if delta < -1e-12:raise ValueError('DYNAMIC_BELOW_SAMPLE_BEST')
  pairs.append({'task':r['task'],'dynamic_minus_samplebest':delta})
 return {'N':len(rows),'means':means,'dynamic_minus_sample_best_pp':100*(means['DYNAMIC']-means['SAMPLE_BEST']),
         'bootstrap':formal_dynamic_bootstrap(pairs,list(counts),means['DYNAMIC'],means['SAMPLE_BEST'],seed,resamples),
         'row_order':'RELEASED_CSV_ORDER','task_order':list(counts)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 p.add_argument('--seed',type=int,default=20260828);p.add_argument('--resamples',type=int,default=10000);x=p.parse_args()
 if x.resamples<1:p.error('resamples must be positive')
 with Path(x.input).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 Path(x.output).write_text(json.dumps(analyze(rows,x.seed,x.resamples),indent=2,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
