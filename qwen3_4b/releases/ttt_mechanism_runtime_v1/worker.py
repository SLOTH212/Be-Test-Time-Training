"""Sample-exclusive, crash-resumable execution with immutable configuration identity."""
import contextlib,fcntl,hashlib,json,os,time
from pathlib import Path
from runtime import canonical

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name('.'+path.name+f'.{os.getpid()}.{time.time_ns()}')
 try:
  with temp.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(temp,path);fd=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if temp.exists():temp.unlink()
class Store:
 def __init__(self,root,identity):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.identity=canonical(identity)
  # The lock inode is permanent. Kernel flock releases ownership on process exit.
  with (self.root/'config.lock').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   p=self.root/'CONFIG_IDENTITY.json'
   if p.exists():
    if json.loads(p.read_text())['sha256']!=self.identity:raise ValueError('CONFIG_IDENTITY_MISMATCH')
   else:atomic(p,{'sha256':self.identity,'config':identity})
 def path(self,sid):return self.root/'samples'/(hashlib.sha256(sid.encode()).hexdigest()+'.json')
 def committed(self,sid):
  p=self.path(sid)
  if not p.exists():return None
  x=json.loads(p.read_text())
  if x.get('config_authority')!=self.identity or x.get('sample_id')!=sid or x.get('status')!='COMMITTED':raise ValueError('INVALID_COMMIT')
  h=x.get('payload_sha256');body={k:v for k,v in x.items() if k!='payload_sha256'}
  if h!=canonical(body):raise ValueError('COMMIT_HASH_MISMATCH')
  return x
 @contextlib.contextmanager
 def own(self,sid):
  d=self.root/'locks';d.mkdir(exist_ok=True)
  with (d/(hashlib.sha256(sid.encode()).hexdigest()+'.lock')).open('a+') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('SAMPLE_OWNED_BY_ANOTHER_WORKER')
   try:yield
   finally:fcntl.flock(f,fcntl.LOCK_UN)
 def commit(self,sid,payload):
  # Caller must hold own(sid) across check, execution, and commit.
  if self.committed(sid) is not None:raise ValueError('DUPLICATE_COMMIT')
  x={**payload,'sample_id':sid,'config_authority':self.identity,'status':'COMMITTED'};x['payload_sha256']=canonical(x);atomic(self.path(sid),x)
