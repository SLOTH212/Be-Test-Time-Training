import numpy as np
import statistics
from collections import defaultdict
CELLS=["F00","F10","F01","F11"]
LEVELS=["L1","L2","L3","L4","L5","L6"]

def describe(x):
 a=np.asarray(x,float);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'std':float(a.std(ddof=1)) if len(a)>1 else 0.0,'q05':float(np.quantile(a,.05)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'q95':float(np.quantile(a,.95))}


def boot(x,seed=20260827,B=20000):
 a=np.asarray(x,float);rng=np.random.default_rng(seed);v=np.empty(B)
 for i in range(B):v[i]=a[rng.integers(0,len(a),len(a))].mean()
 return {'mean':float(a.mean()),'median':float(np.median(a)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'replicates':B,'seed':seed,'paired':True}


def effects(d):
 return {'K_AT_P0':d['F00']-d['F10'],'K_AT_P1':d['F01']-d['F11'],'P_AT_K0':d['F00']-d['F01'],'P_AT_K1':d['F10']-d['F11'],'MAIN_K':.5*((d['F00']-d['F10'])+(d['F01']-d['F11'])),'MAIN_P':.5*((d['F00']-d['F01'])+(d['F10']-d['F11'])),'INTERACTION':d['F00']-d['F10']-d['F01']+d['F11']}
