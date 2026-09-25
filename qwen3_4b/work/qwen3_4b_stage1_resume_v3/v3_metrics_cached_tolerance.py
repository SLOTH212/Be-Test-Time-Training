"""Read-only, full-element V3 final-state comparisons; CPU only."""
import os, json, math, hashlib, itertools, time
from pathlib import Path
import numpy as np
import torch

R = Path('/path/to/ttt')
A = R / 'audits/v3'
RUN = R / 'runs/qwen3_4b_stage1_resume_v3/attempt2'
OUT = A / 'analysis'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
assert not torch.cuda.is_initialized()
names = [f'{c}{i}' for c in 'UR' for i in range(1, 6)]
receipts = json.loads((A / 'execution2/REPLICATE_CONTROLLER_RECEIPTS.json').read_text())
valid = lambda n: all(any(x['replica']==n and x['role']==role and x['status']=='VALID_COMPLETED_PHASE' for x in receipts) for role in (['U'] if n[0]=='U' else ['R_PRE','R_POST']))
names = [n for n in names if valid(n)]
assert sum(n[0]=='U' for n in names)>=4 and sum(n[0]=='R' for n in names)>=4
pairs = list(itertools.combinations(range(len(names)), 2))
inventory = json.loads((A / 'V2_FAILED_76_TENSOR_INVENTORY.json').read_text())['rows']
primary = {row['group'] + row['canonical_state_name'] for row in inventory}
meta = {n:[json.loads((RUN/n/f'final_rank{k}.json').read_text()) for k in (0,1)] for n in names}
data = {n:[torch.load(RUN/n/f'final_rank{k}.pt', mmap=True, weights_only=True, map_location='cpu') for k in (0,1)] for n in names}
keys = sorted(data[names[0]][0])
assert len(keys)==1647 and len(primary)==76 and primary.issubset(keys)
for n in names:
 for k in (0,1):
  assert set(data[n][k])==set(keys) and meta[n][k]['status']=='PASS'
manifest = json.loads((OUT/'PAIR_METRIC_MANIFEST.json').read_text())
amendment = json.loads((A/'METRICS_CPU_EQUIVALENT_OPTIMIZATION.json').read_text())
assert hashlib.sha256((OUT/'PAIR_METRIC_MANIFEST.json').read_bytes()).hexdigest()==amendment['verified_raw_manifest_sha256']
assert manifest['names']==names and len(manifest['source_files'])==len(names)*2
for item in manifest['source_files']:
 p=Path(item['path']);assert p.stat().st_size==item['bytes']
 assert item['sha256']==meta[p.parent.name][int(p.stem[-1])]['sha256']
print('REUSE_ALREADY_COMPLETED_FULL_FILE_SHA256_VERIFICATION',len(manifest['source_files']),flush=True)

def quantiles(hist, n):
 vals=np.array(sorted(hist),dtype=np.float64);counts=np.array([hist[v] for v in vals],dtype=np.int64);cum=counts.cumsum()
 assert int(cum[-1])==n
 result={}
 for q in (.5,.95,.99,1.):
  pos=q*(n-1);lo=math.floor(pos);hi=math.ceil(pos)
  a=vals[np.searchsorted(cum,lo+1)];b=vals[np.searchsorted(cum,hi+1)]
  result[str(q)]=float(a+(b-a)*(pos-lo))
 return result

target=OUT/'ALL_TENSOR_PAIR_METRICS.jsonl'
completed=[json.loads(line) for line in target.read_text().splitlines()] if target.exists() else []
assert [x['key'] for x in completed]==keys[:len(completed)]
completed_by_key={x['key']:x for x in completed}
with target.open('a') as output:
 for ki,key in enumerate(keys):
  if 0<ki<len(completed):continue
  m=meta[names[0]][0]['tensors'][key];rep=m['replicated'];shape=m['global_shape'];n=math.prod(shape)
  group='model' if key.startswith('model.') else 'optimizer';atol=1e-5 if group=='model' else 1e-6
  shards=[0] if rep else [0,1]
  for name in names:
   assert sum(data[name][r][key].numel() for r in shards)==n
   for rank in (0,1):
    mm=meta[name][rank]['tensors'][key]
    assert mm['global_shape']==shape and mm['dtype']==m['dtype'] and mm['replicated']==rep
   if rep:assert torch.equal(data[name][0][key],data[name][1][key])
  sums=np.zeros((len(names),4),np.float64) # square, abs, signed, maximum
  ps=np.zeros((len(pairs),7),np.float64) # max, sumabs, FP64 square, nonzero, isclose violations right/reverse, legacy FP32 square
  hist=[{} for _ in pairs] if key in primary else None
  for rank in shards:
   arrays=[data[name][rank][key].view(-1) for name in names]
   for start in range(0,arrays[0].numel(),1<<20):
    xs=[x[start:start+(1<<20)].float().numpy() for x in arrays]
    allowed=[((torch.from_numpy(x)*.01).abs()+atol).numpy() for x in xs]
    for i,x in enumerate(xs):
     assert np.isfinite(x).all(),('NONFINITE',key,names[i])
     xd=x.astype(np.float64);sums[i,0]+=np.dot(xd,xd);sums[i,1]+=np.abs(xd).sum();sums[i,2]+=xd.sum();sums[i,3]=max(sums[i,3],float(np.abs(x).max(initial=0)))
    for p,(i,j) in enumerate(pairs):
     diff=np.abs(xs[i]-xs[j]);dd=diff.astype(np.float64)
     ps[p,0]=max(ps[p,0],float(diff.max(initial=0)));ps[p,1]+=dd.sum();ps[p,2]+=np.dot(dd,dd);ps[p,3]+=np.count_nonzero(diff)
     ps[p,4]+=np.count_nonzero(diff>allowed[j])
     ps[p,5]+=np.count_nonzero(diff>allowed[i])
     ps[p,6]+=np.square(diff).sum(dtype=np.float64)
     if hist is not None:
      vals,counts=np.unique(diff,return_counts=True)
      for v,c in zip(vals,counts):hist[p][float(v)]=hist[p].get(float(v),0)+int(c)
  norms=np.sqrt(sums[:,0]);scale=max(float(np.median([norms[i] for i,name in enumerate(names) if name[0]=='U'])),atol*math.sqrt(n),np.finfo(float).tiny)
  row={'key':key,'group':group,'primary76':key in primary,'shape':shape,'numel':n,'dtype':m['dtype'],'replicated':rep,'atol':atol,'rtol':.01,'U_relative_scale':scale,'run_stats':{},'pairs':[]}
  for i,name in enumerate(names):row['run_stats'][name]={'norm':float(norms[i]),'max_abs':float(sums[i,3]),'mean_abs':float(sums[i,1]/n),'signed_mean':float(sums[i,2]/n),'RMS':float(norms[i]/math.sqrt(n))}
  for p,(i,j) in enumerate(pairs):
   mx,l1,sq,nz,viol,reverse,legacy_sq=map(float,ps[p]);l2=math.sqrt(sq)
   d={'left':names[i],'right':names[j],'class':names[i][0]+names[j][0],'max_abs':mx,'mean_abs':l1/n,'L1':l1,'L2':l2,'relative_L2':l2/scale,'pair_reference_relative_L2':l2/max(norms[i],np.finfo(float).tiny),'exact':nz==0,'old_allclose':viol==0,'old_allclose_reverse':reverse==0,'fraction_differing':nz/n,'fraction_old_violations':viol/n,'signed_mean_right_minus_left':float((sums[j,2]-sums[i,2])/n)}
   d['legacy_rms_float32_square']=math.sqrt(legacy_sq/n)
   if hist is not None:d['absolute_element_quantiles']=quantiles(hist[p],n)
   row['pairs'].append(d)
  if key in completed_by_key:
   assert row==completed_by_key[key],('OPTIMIZATION_NOT_BITWISE_IDENTICAL_TO_ORIGINAL_FULL_TENSOR',key)
   print('FULL_TENSOR_ALL45_METRICS_BITWISE_MATCH_ORIGINAL',key,flush=True)
   continue
  output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
  if ki%10==0:print('TENSOR',ki+1,len(keys),key,flush=True)
(OUT/'PAIR_METRICS_COMPLETE.json').write_text(json.dumps({'status':'PASS','tensor_count':len(keys),'primary_count':len(primary),'pairs':len(pairs),'names':names,'completed_at':time.time(),'metrics_sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2)+'\n')
print('FULL_METRICS_PASS',flush=True)
