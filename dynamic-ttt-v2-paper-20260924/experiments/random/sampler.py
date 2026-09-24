"""Reward-independent, uniform nonconstant sampling without replacement."""
import hashlib,json
import numpy as np
MASTER=20260923
ACTIONS=['OFF','L0','L6','L12','L18','L24','ALL']
def seed(sid):return int.from_bytes(hashlib.sha256(f'{MASTER}|replicate=0|{sid}'.encode()).digest()[:8],'big')
def decode(n,K,T):
 out=[0]*T
 for i in range(T-1,-1,-1):n,out[i]=divmod(n,K)
 assert n==0
 return out
def encode(ds,K):
 n=0
 for d in ds:n=n*K+d
 return n
def below(rng,n):
 if n<=np.iinfo(np.int64).max:return int(rng.integers(0,n))
 bits=(n-1).bit_length();mask=(1<<bits)-1
 while True:
  x=int.from_bytes(rng.bytes((bits+7)//8),'big')&mask
  if x<n:return x

def sample(sid,T,B,actions=ACTIONS):
 K=len(actions);space=K**T;M=space-K;assert T>=2 and B>0 and M>0
 rng=np.random.default_rng(seed(sid));seen=set();out=[];reject_constant=reject_duplicate=0
 if B>=M:
  draws=(x for x in range(space) if len(set(decode(x,K,T)))>1)
  out=list(draws)
 else:
  while len(out)<B:
   x=below(rng,space)
   if len(set(decode(x,K,T)))==1:reject_constant+=1;continue
   if x in seen:reject_duplicate+=1;continue
   seen.add(x);out.append(x)
 return [dict(sample_id=sid,sample_seed=seed(sid),trajectory_index=i,trajectory=[actions[d] for d in decode(x,K,T)],trajectory_hash=hashlib.sha256(json.dumps([actions[d] for d in decode(x,K,T)],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),T=T,B_i=B) for i,x in enumerate(out)],dict(space_size=str(M),space_exhausted=B>=M,rejected_constant=reject_constant,rejected_duplicate=reject_duplicate)

def render(inputs):
 records=[]
 for r in sorted(inputs,key=lambda x:x['sample_id']):
  xs,_=sample(r['sample_id'],r['T'],r['B_i']);records.extend(xs)
 return ''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in records).encode()
