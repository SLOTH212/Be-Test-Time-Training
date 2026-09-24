from pathlib import Path
import json,csv,hashlib,ast,difflib
O=Path(__file__).resolve().parent
def j(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
pre=j(O/'PRE_FLIGHT_V2.json');protocol=j(O/'FROZEN_V2_STAGE1_PROTOCOL.json');s=j(O/'V2_STAGE1_SUMMARY.json');checks={};hash_checks=[]
for path,expected in {**pre['source_hashes'],**protocol['source_files'],str(O/'formal.py'):protocol['formal_driver_sha256'],str(O/'V2_STAGE1_COHORT.json'):protocol['cohort_sha256']}.items():
 actual=sha(path);hash_checks.append({'path':path,'expected':expected,'actual':actual,'match':actual==expected})
checks['historical_and_frozen_sources_unchanged']=all(x['match'] for x in hash_checks)
expected={'V2_FIXED7_173.csv':1211,'V2_NATIVE_173.csv':173,'V2_SAMPLE_BEST_173.csv':173,'V2_GAP_173.csv':173,'V2_HISTORICAL_SCORE_SHIFT.csv':8,'V2_STATE_ISOLATION.csv':1384}
for n,num in expected.items():checks['count:'+n]=len(list(csv.DictReader((O/n).open())))==num
r=[json.loads(l) for l in (O/'V2_TRAJECTORIES.jsonl').open()];checks['unique_complete_cells']=len(r)==len({(x['sample_id'],x['action']) for x in r})==1384;checks['all_native_identity']=s['formal_replay']['native_identity_match_n']==173;checks['all_state_checks']=all(x['checkpoint_unchanged'] and x['state_reset_pass'] for x in r);checks['all_five_functional']=all(x['materialized_layers']==x['functional_layers']==[0,6,12,18,24] for x in r);checks['precheck_gate']=s['precheck']['status']=='PASS_EXACT';checks['no_errors']=s['formal_replay']['error_n']==0
AL={'OFF':[],'ALL':[0,6,12,18,24],**{'L'+str(l):[l] for l in [0,6,12,18,24]}}
checks['actual_update_event_counts']=all(x['layer_update_counts']=={str(l):sum(l in AL[a] for a in x['sequence']) for l in [0,6,12,18,24]} for x in r)
checks['no_tail_or_generation_writes']=all(x['tail_update_count']==x['generation_update_count']==0 for x in r)
# Independent standard-library reconstruction of core counts and means.
by={(x['sample_id'],x['action']):x['score'] for x in r};ids=sorted({x['sample_id'] for x in r});A=['OFF','L0','L6','L12','L18','L24','ALL'];g=[100*(by[s,'NATIVE']-max(by[s,a] for a in A)) for s in ids];tol=protocol['formal_tolerance']*100;checks['independent_GT']=sum(v>tol for v in g)==s['core_result']['GT'];checks['independent_EQ']=sum(abs(v)<=tol for v in g)==s['core_result']['EQ'];checks['independent_LT']=sum(v< -tol for v in g)==s['core_result']['LT'];checks['independent_mean']=abs(sum(g)/173-s['core_result']['gap_pp']['mean'])<1e-12
for p in O.glob('*.py'):ast.parse(p.read_text())
forbidden={'raw_prompt','raw_reference','raw_prediction','prompt','input','reference','prediction','answer','generated_text'};violations=[]
def walk(x,path):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in forbidden:violations.append({'path':path,'key':k})
   walk(v,path)
 elif isinstance(x,list):
  for v in x:walk(v,path)
for p in O.rglob('*.json'):walk(j(p),str(p))
for x in r:walk(x,'V2_TRAJECTORIES.jsonl')
for p in O.glob('*.csv'):
 for k in next(csv.reader(p.open())):
  if k in forbidden:violations.append({'path':str(p),'key':k})
checks['publication_safe_fields']=not violations;checks['no_tensor_dumps']=not any(p.suffix in ['.pt','.pth','.npz','.npy','.safetensors'] for p in O.rglob('*') if p.is_file())
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'hash_checks':hash_checks,'publication_safe_field_violations':violations,'scientific_result_not_changed_by_verifier':True};(O/'V2_POSTFLIGHT_PACKAGE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
files=sorted(p for p in O.rglob('*') if p.is_file() and p.name!='SHA256SUMS.generated.txt');(O/'SHA256SUMS.generated.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(O))+'\n' for p in files));print(json.dumps({'status':result['status'],'failed':[k for k,v in checks.items() if not v],'hash_check_n':len(hash_checks),'artifact_n':len(files),'total_bytes':sum(p.stat().st_size for p in files),'violations':violations}))
