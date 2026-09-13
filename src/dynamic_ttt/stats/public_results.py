"""Apply preserved estimators to the sanitized per-sample CSV surface.

Row order is the released CSV order. Finite-bootstrap endpoints can differ from
historical tables if their original aggregation used a different row order.
"""
import argparse,csv,json,statistics
from pathlib import Path
from dynamic_ttt.stats import reverse,reset,deletion,kp,r2

def analyze(rows,mechanism,seeds):
 if not rows:raise ValueError('EMPTY_RESULTS')
 B=seeds['resamples']
 out={'mechanism':mechanism,'statistical_procedure':seeds,'row_order':'RELEASED_CSV_ORDER'}
 if mechanism in ['reverse','reset','deletion']:
  if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_SAMPLE')
  effects=[float(r['native'])-float(r['counterfactual']) for r in rows]
  m={'reverse':reverse,'reset':reset,'deletion':deletion}[mechanism]
  key={'reverse':'dynamic_minus_reverse','reset':'native_minus_reset','deletion':'normal_minus_selective'}[mechanism]
  rs=[{'task':r['task'],key:e} for r,e in zip(rows,effects)]
  strat=m.stratified_ci if mechanism=='reverse' else m.strat
  out.update(N=len(rows),mean_effect=statistics.fmean(effects),paired_ci=m.ci(effects,seeds['paired'],B),task_stratified_ci=strat(rs,seeds['stratified'],B))
  if mechanism=='deletion':
   positive=[(e,float(r['native'])-float(r['full_reset'])) for r,e in zip(rows,effects) if float(r['native'])>float(r['full_reset'])]
   out.update(positive_denominator_n=len(positive),nonpositive_denominator_n=len(rows)-len(positive),mean_explained_fraction=statistics.fmean(e/d for e,d in positive) if positive else None,aggregate_explained_fraction=sum(e for e,d in positive)/sum(d for e,d in positive) if positive else None)
 elif mechanism=='r2':
  if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_SAMPLE')
  rs=[{'task':r['task'],'Dynamic_minus_R2':float(r['native_reward'])-float(r['r2_reward']),'Dynamic_minus_OFF':float(r['native_reward'])-float(r['off_reward']),'R2_minus_OFF':float(r['r2_reward'])-float(r['off_reward'])} for r in rows]
  out.update(N=len(rows),paired_bootstrap=r2.bootstrap(rs,seeds['paired'],B,False),task_stratified_bootstrap=r2.bootstrap(rs,seeds['stratified'],B,True))
 elif mechanism=='kp':
  groups={}
  for r in rows:
   cells=groups.setdefault(r['sample_id'],{})
   if r['cell'] in cells:raise ValueError('DUPLICATE_CELL')
   cells[r['cell']]=float(r['distance_generation_start_relative_l2'])
  if any(set(c)!=set(kp.CELLS) for c in groups.values()):raise ValueError('INCOMPLETE_FACTORIAL')
  effects=[kp.effects(c) for c in groups.values()]
  out.update(N=len(groups),paired_factorial_bootstrap={k:kp.boot([e[k] for e in effects],seeds['paired'],B) for k in effects[0]})
 else:raise ValueError('UNKNOWN_MECHANISM')
 return out

def main(mechanism=None):
 p=argparse.ArgumentParser(description=__doc__)
 if mechanism is None:p.add_argument('--mechanism',choices=['reverse','reset','deletion','kp','r2'],required=True)
 p.add_argument('--input',required=True);p.add_argument('--seeds',required=True);p.add_argument('--output',required=True);x=p.parse_args()
 seeds=json.loads(Path(x.seeds).read_text(encoding='utf-8'))
 if type(seeds.get('resamples')) is not int or seeds['resamples']<1:raise ValueError('INVALID_RESAMPLES')
 with Path(x.input).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 result=analyze(rows,mechanism or x.mechanism,seeds)
 Path(x.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
