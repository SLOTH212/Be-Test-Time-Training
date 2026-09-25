import json,hashlib,copy,time,ast,importlib.util
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
f=read(A/'FINAL_REQUIRED_FIELDS.json');assert f['V4_CLASSIFICATION']=='SHARED_BOUNDARY_RESUME_CONSISTENT' and f['V4_INDEPENDENT_AUDIT']=='PASS';assert read(A/'INDEPENDENT_AUDIT.json')['status']=='PASS'
v2p=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V2.json';v2=read(v2p);assert sha(v2p)=='7138c1bc6cd30892b152224e71d36ee594eeb4451ca4af52ef4654998a36ad52'
for k in ['REAL_ONE_TOKEN_GPU_EDGE','REAL_ODD_FINAL_GPU_EDGE','REAL_DATA_FAILURE_RECOVERY','CHECKPOINT_STORAGE_PREFLIGHT','CHECKPOINT_RETENTION_RUNTIME']:assert v2[k]=='PASS'
authpath=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V4.json';guardpath=R/'bin/qwen3_4b_stage1_gpu23_formal_control_v4.py';assert not authpath.exists() and not guardpath.exists()
source=Path(v2['V2_LAUNCH_GUARD_PATH']).read_text();start=source.index('HARD_V2_GATES=');end=source.index('\ndef guard():',start)
validator=r"""HARD_V4_GATES=['TRAINING_RUNTIME_AUTHORITY','MODEL_AUTHORITY_STATUS','STAGE1_DATASET_RUNTIME_AUTHORITY','REAL_ONE_TOKEN_GPU_EDGE','REAL_ODD_FINAL_GPU_EDGE','FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING','REAL_DATA_FAILURE_RECOVERY','CHECKPOINT_STORAGE_PREFLIGHT','CHECKPOINT_RETENTION_RUNTIME','SHARED_BOUNDARY_IDENTITY','SAVE_OPERATION_STATE_NONMUTATION','SHARED_BOUNDARY_RNG_PARITY','B1_POSTLOAD_STATE_EXACT','B2_POSTLOAD_STATE_EXACT','DISTRIBUTED_REDUCTION_CORRECT_A','DISTRIBUTED_REDUCTION_CORRECT_B1','DISTRIBUTED_REDUCTION_CORRECT_B2','GRADIENT_CLIP_OPERATOR_CORRECT','V4_INDEPENDENT_AUDIT','ORIGINAL_MODE_RESUME_STATE_INTEGRITY','ORIGINAL_MODE_RESUME_CAUSAL_CHECK']
def validate_v4_authority(auth):
 if auth.get('PRELAUNCH_AUTHORITY_VERSION')!='V4':raise RuntimeError('V4_AUTHORITY_REQUIRED')
 if auth.get('V4_CLASSIFICATION')!='SHARED_BOUNDARY_RESUME_CONSISTENT' or auth.get('SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23')!='YES':raise RuntimeError('V4_CAUSAL_GATE_NOT_PASS')
 for key in HARD_V4_GATES:
  if auth.get(key)!='PASS':raise RuntimeError('V4_HARD_GATE_FAILED '+key)
 if auth.get('LEGACY_V2_NUMERICAL_PARITY')!='FAIL' or auth.get('V3_CLASSIFICATION_PRESERVED')!='INCONCLUSIVE':raise RuntimeError('HISTORICAL_RESULT_ERASED')
 if auth.get('V4_PROTOCOL_FROZEN_BEFORE_GPU') is not True or auth.get('MISSING_MUTABLE_STATE_N')!=0:raise RuntimeError('V4_EVIDENCE_CONTRACT')
 if auth.get('SCIENTIFIC_SEMANTICS_CHANGED') is not False or auth.get('FORMAL_MODE')!='ORIGINAL' or auth.get('FORMAL_DETERMINISTIC_MODE_ENABLED') is not False:raise RuntimeError('ORIGINAL_SCIENTIFIC_MODE_REQUIRED')
 if auth.get('PHYSICAL_GPU_ALLOWLIST')!=[2,3] or auth.get('UNAUTHORIZED_GPU_USED') is not False:raise RuntimeError('V4_GPU_POLICY')
 for key,path in [('V4_FINAL_AUDIT_SHA256','V4_FINAL_AUDIT_PATH'),('V4_PROTOCOL_SHA256','V4_PROTOCOL_PATH'),('V3_FINAL_AUDIT_SHA256','V3_FINAL_AUDIT_PATH'),('V2_AUTHORITY_SHA256','V2_AUTHORITY_PATH')]:
  if not auth.get(key) or digest(auth[path])!=auth[key]:raise RuntimeError('V4_EVIDENCE_HASH_MISMATCH '+key)
 final=json.loads(pathlib.Path(auth['V4_FINAL_AUDIT_PATH']).read_text())
 if final['classification']!='SHARED_BOUNDARY_RESUME_CONSISTENT' or final['decision']['failed_observable_n']!=0:raise RuntimeError('V4_AUDIT_NOT_CONSISTENT')
 independent=json.loads(pathlib.Path(final['independent_audit_path']).read_text())
 if digest(final['independent_audit_path'])!=final['independent_audit_sha256'] or independent['status']!='PASS':raise RuntimeError('V4_INDEPENDENT_AUDIT_INVALID')
 if auth.get('GPU_FOREIGN_PROCESS_POLICY')!='WARNING_ONLY_NO_PROCESS_MANIPULATION':raise RuntimeError('V4_SHARING_POLICY')
"""
source=source[:start]+validator+source[end:];source=source.replace('PRELAUNCH_AUTHORITY_V2.json','PRELAUNCH_AUTHORITY_V4.json').replace('validate_v2_authority(auth)','validate_v4_authority(auth)')
old='source=json.loads(pathlib.Path(cfg["runtime_source_manifest"]).read_text())\n for rel,expected in source.items():\n  if digest(ROOT/"src/training_runtime"/rel)!=expected:raise RuntimeError("RUNTIME_SOURCE_HASH_MISMATCH "+rel)'
new='source=json.loads(pathlib.Path(auth["RUNTIME_SOURCE_MANIFEST_PATH"]).read_text())\n for path,expected in source.items():\n  if digest(path)!=expected:raise RuntimeError("RUNTIME_SOURCE_HASH_MISMATCH "+path)';assert old in source;source=source.replace(old,new);compile(source,str(guardpath),'exec');guardpath.write_text(source)
auth=dict(v2);auth.update(f);auth.update(PRELAUNCH_AUTHORITY_VERSION='V4',schema='QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V4',created_at=time.time(),QWEN3_4B_STAGE1_PRELAUNCH_STATUS='PASS_WITH_WARNING',V2_AUTHORITY_PATH=str(v2p),V2_AUTHORITY_SHA256=sha(v2p),V3_FINAL_AUDIT_PATH=str(R/'audits/qwen3_4b_stage1_resume_equivalence_v3/FINAL_RESUME_EQUIVALENCE_AUDIT.json'),V4_LAUNCH_GUARD_PATH=str(guardpath),V4_LAUNCH_GUARD_SHA256=sha(guardpath),FORMAL_RUNTIME_ENTRYPOINT=v2['FORMAL_RUNTIME_ENTRYPOINT'],ORIGINAL_MODE_RESUME_NUMERICAL_PARITY='FAIL',FORMAL_STAGE1_TRAINING_STARTED=False)
auth['supersession']='V4 shared-boundary causal evidence supersedes the launch blocker, preserving V2 numerical FAIL and V3 INCONCLUSIVE. It does not rewrite those authorities.'
auth['inherited_v2_status']='FAIL_PRESERVED_IN_V2_AUTHORITY';auth.pop('BLOCKING_PHASE',None);auth.pop('BLOCKING_REASON',None)
auth['file_sha256']=dict(v2['file_sha256'])
for p in [v2p,Path(auth['V3_FINAL_AUDIT_PATH']),Path(f['V4_FINAL_AUDIT_PATH']),Path(f['V4_PROTOCOL_PATH']),A/'INDEPENDENT_AUDIT.json',A/'SHARED_DCP_IDENTITY.json',guardpath]:auth['file_sha256'][str(p)]=sha(p)
# Bind every referenced final audit source/receipt and immutable shared DCP manifest.
for p,h in read(Path(f['V4_FINAL_AUDIT_PATH']))['evidence_files'].items():auth['file_sha256'][p]=h
for p,h in auth['file_sha256'].items():assert sha(Path(p))==h,('AUTHORITY_BOUND_FILE_CHANGED',p)
authpath.write_text(json.dumps(auth,ensure_ascii=False,indent=2)+'\n');authpath.with_suffix('.sha256').write_text(sha(authpath)+'  '+authpath.name+'\n')
# CPU-only guard contract verification. Import does not invoke guard()/launch().
spec=importlib.util.spec_from_file_location('v4_guard_verification',guardpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.validate_v4_authority(auth);tests=[]
for key,value in [('PRELAUNCH_AUTHORITY_VERSION','V2'),('V4_CLASSIFICATION','INCONCLUSIVE'),('LEGACY_V2_NUMERICAL_PARITY','PASS'),('V3_CLASSIFICATION_PRESERVED','PASS'),('MISSING_MUTABLE_STATE_N',1),('PHYSICAL_GPU_ALLOWLIST',[0,1])]+[(k,'FAIL') for k in module.HARD_V4_GATES]:
 bad=copy.deepcopy(auth);bad[key]=value
 try:module.validate_v4_authority(bad)
 except (RuntimeError,KeyError):tests.append({'mutated_field':key,'rejected':True})
 else:raise AssertionError('GUARD_ACCEPTED_INVALID '+key)
receipt={'DEBUG_ONLY':True,'status':'PASS','scope':'CPU-only real authority validator and negative gate checks; launch/supervise not called. Existing model/data/environment/GPU runtime checks retained for a future authorized launch.','guard_sha256':sha(guardpath),'authority_sha256':sha(authpath),'tests':tests,'formal_training_started':False};(A/'V4_LAUNCH_GUARD_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
f.update(PRELAUNCH_AUTHORITY_V4_PATH=str(authpath),PRELAUNCH_AUTHORITY_V4_SHA256=sha(authpath),V4_LAUNCH_GUARD_PATH=str(guardpath),V4_LAUNCH_GUARD_SHA256=sha(guardpath),V4_LAUNCH_GUARD_VERIFICATION='PASS');(A/'FINAL_REQUIRED_FIELDS.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,indent=2))
