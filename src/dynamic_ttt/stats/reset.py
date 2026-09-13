import numpy as np
import statistics
from collections import defaultdict
CELLS=["F00","F10","F01","F11"]
LEVELS=["L1","L2","L3","L4","L5","L6"]

def mean(x): return statistics.fmean(x)


def ci(x,seed,n):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); y=np.empty(n)
 for s in range(0,n,1000): k=min(1000,n-s); y[s:s+k]=x[rng.integers(0,len(x),(k,len(x)))].mean(1)
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]


def strat(rs,seed,n):
 g=defaultdict(list)
 for r in rs:g[r["task"]].append(r["native_minus_reset"])
 rng=np.random.default_rng(seed); y=np.zeros(n); total=len(rs)
 for v in g.values():
  x=np.asarray(v,float)
  for s in range(0,n,1000): k=min(1000,n-s); y[s:s+k]+=x[rng.integers(0,len(x),(k,len(x)))].mean(1)*len(x)/total
 return [float(np.quantile(y,.025)),float(np.quantile(y,.975))]
