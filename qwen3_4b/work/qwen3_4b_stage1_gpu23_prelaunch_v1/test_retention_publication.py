import pathlib,json,hashlib,os,shutil,importlib.util
ROOT=pathlib.Path('/path/to/ttt')
WORK=ROOT/'work/qwen3_4b_stage1_gpu23_prelaunch_v1'
AUDIT=ROOT/'audits/qwen3_4b_stage1_gpu23_prelaunch_corrected_v2'
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fsync_dir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic_json(p,data):
 t=p.with_suffix('.testtmp');t.write_text(json.dumps(data))
 os.replace(t,p);fsync_dir(p.parent)
helpers=(WORK/'checkpoint_retention_helpers.py').read_text()
def make(root,name,generation):
 p=root/name;p.mkdir()
 (p/'payload').write_text(str(generation))
 (p/'manifest.json').write_text(json.dumps({'complete':True,'slot':name.split('.')[0],'files':{'payload':digest(p/'payload')},'progress':{'step':generation}}))
 return p
def generation(p):return int((p/'payload').read_text()) if p.exists() else None
class Fault(Exception):pass
results=[]
for case in ['partial_write','before_backup','after_backup','after_replacement','after_pointer','recovery_interrupted','successful_commit','invalid_pointer','first_uncommitted','corrupt_partial','invalid_pointer_without_backup']:
 root=WORK/'retention_fault_fixtures_validated_v2'/case
 root.mkdir(parents=True,exist_ok=False)
 ns=dict(json=json,os=os,shutil=shutil,_digest=digest,fsync_dir=fsync_dir,atomic_json=atomic_json)
 exec(helpers,ns)
 if case=='first_uncommitted':
  make(root,'slot_A.partial',1)
  try:ns['_recover_publication'](root)
  except RuntimeError as e:assert str(e)=='DCP_UNCOMMITTED_ARTIFACTS_NO_LATEST'
  else:raise AssertionError(case)
  assert (root/'slot_A.partial').exists()
  results.append({'case':case,'status':'PASS'});continue
 a=make(root,'slot_A',1);b=make(root,'slot_B',2);partial=make(root,'slot_A.partial',3)
 atomic_json(root/'latest.json',{'slot':'slot_B','progress':{'step':2},'manifest_sha256':digest(b/'manifest.json')})
 if case=='partial_write':
  ns['_recover_publication'](root)
  assert generation(a)==1 and generation(b)==2
 elif case=='corrupt_partial':
  (partial/'payload').write_text('CORRUPTED')
  try:ns['_publish_checkpoint'](partial,a,{'step':3})
  except RuntimeError as e:assert str(e).startswith('DCP_FILE_HASH_MISMATCH')
  else:raise AssertionError(case)
  assert generation(a)==1 and generation(b)==2
 elif case=='invalid_pointer_without_backup':
  (root/'latest.json').write_text('{}')
  try:ns['_recover_publication'](root)
  except RuntimeError:pass
  else:raise AssertionError(case)
  assert generation(a)==1 and generation(b)==2
 elif case=='invalid_pointer':
  os.replace(a,root/'slot_A.previous')
  (root/'latest.json').write_text('{}')
  try:ns['_recover_publication'](root)
  except RuntimeError:pass
  else:raise AssertionError(case)
  assert generation(root/'slot_A.previous')==1 and generation(b)==2
 else:
  replace=os.replace
  original_atomic=ns['atomic_json']
  def inject_replace(src,dst):
   if case=='before_backup' and str(dst).endswith('.previous'):raise Fault(case)
   replace(src,dst)
   if case in ('after_backup','recovery_interrupted') and str(dst).endswith('.previous'):raise Fault(case)
   if case=='after_replacement' and str(src).endswith('.partial'):raise Fault(case)
  def inject_atomic(p,data):
   original_atomic(p,data)
   if case=='after_pointer':raise Fault(case)
  os.replace=inject_replace
  ns['atomic_json']=inject_atomic
  try:
   ns['_publish_checkpoint'](partial,a,{'step':3})
   assert case=='successful_commit'
  except Fault:assert case!='successful_commit'
  finally:os.replace=replace;ns['atomic_json']=original_atomic
  committed=json.loads((root/'latest.json').read_text())['slot']=='slot_A'
  assert generation(b)==2
  assert generation(a if a.exists() and generation(a)==1 else root/'slot_A.previous')==1 or committed
  if case=='recovery_interrupted':
   removal=ns['_remove_retention_artifact']
   def interrupted_remove(p,r):
    removal(p,r);raise Fault('during recovery')
   ns['_remove_retention_artifact']=interrupted_remove
   try:ns['_recover_publication'](root)
   except Fault:pass
   ns['_remove_retention_artifact']=removal
  ns['_recover_publication'](root);ns['_recover_publication'](root)
  if committed:assert generation(a)==3 and generation(b)==2
  else:
   assert generation(a)==1 and generation(b)==2
   if partial.exists():shutil.rmtree(partial)
   partial=make(root,'slot_A.partial',3)
   ns['_publish_checkpoint'](partial,a,{'step':3})
   assert generation(a)==3 and generation(b)==2
 results.append({'case':case,'status':'PASS'})
report={'status':'PASS','scope':'process-crash publication/recovery; no hardware power-loss claim','cases':results,'latest_and_previous_retained_until_commit':True,'recovery_idempotent':True,'active_runtime_modified':False}
(AUDIT/'RETENTION_PUBLICATION_FAULT_TEST.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
