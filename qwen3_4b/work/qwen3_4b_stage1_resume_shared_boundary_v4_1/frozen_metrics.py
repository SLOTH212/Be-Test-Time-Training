import math,numpy as np
def metrics(arrays,atol=None):
 n=0;mx=l1=sq=aa=bb=sa=sb=0.;nz=bad=closebad=0
 for a,b in arrays:
  assert a.shape==b.shape and a.dtype==b.dtype
  a=a.reshape(-1);b=b.reshape(-1)
  for off in range(0,a.numel(),1<<20):
   x=a[off:off+(1<<20)].float().numpy();y=b[off:off+(1<<20)].float().numpy();xx=x.astype(np.float64);yy=y.astype(np.float64);dd=np.abs(x-y).astype(np.float64)
   closebad+=int(np.count_nonzero(np.abs(x-y) > (atol+0.01*np.abs(y)))) if atol is not None else 0
   n+=len(x);bad+=int((~np.isfinite(x)).sum()+(~np.isfinite(y)).sum());nz+=int(np.count_nonzero(dd));mx=max(mx,float(dd.max(initial=0)));l1+=float(dd.sum());sq+=float(np.dot(dd,dd));aa+=float(np.dot(xx,xx));bb+=float(np.dot(yy,yy));sa+=float(xx.sum());sb+=float(yy.sum())
 assert not bad,'NONFINITE_CAPTURE'
 return {'numel':n,'exact':nz==0,'max_abs':mx,'mean_abs':l1/max(n,1),'L1':l1,'L2':math.sqrt(sq),'left_norm':math.sqrt(aa),'right_norm':math.sqrt(bb),'left_signed_mean':sa/max(n,1),'right_signed_mean':sb/max(n,1),'fraction_differing':nz/max(n,1),'finite':True,'legacy_allclose':closebad==0 if atol is not None else None,'legacy_mismatch_count':closebad if atol is not None else None}
