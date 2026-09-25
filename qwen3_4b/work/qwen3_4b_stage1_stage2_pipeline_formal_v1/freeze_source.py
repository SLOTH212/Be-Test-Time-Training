import json,time,hashlib
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1';A=R/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=read(R/'configs/qwen3_4b_stage1_stage2_pipeline_formal_v1.yaml')
assert read(A/'CPU_CONTRACT_TESTS.json')['status']=='PASS';assert read(A/'REAL_DEBUG_ARTIFACT_CPU_AUDIT.json')['status']=='PASS'
names=['pipeline_core.py','pipeline_finalize.py','pipeline_supervisor.py','pipeline_prune.py','gpu45_resource_loader.py']
files={str(W/n):sha(W/n) for n in names};files.update(c['bindings'])
for path in [Path(c['supervisor']),R/'configs/qwen3_4b_stage1_stage2_pipeline_formal_v1.yaml']+list((R/'bin').glob('*qwen3_4b*stage1_stage2*formal_v1.sh'))+[R/'bin/finalize_qwen3_4b_stage1_to_stage2_formal_v1.sh',R/'bin/prune_qwen3_4b_intermediate_checkpoints_after_final_v1.sh']:
 files[str(path)]=sha(path)
p=R/'provenance/QWEN3_4B_STAGE1_STAGE2_PIPELINE_SOURCE_AUTHORITY_V1.json'
assert not p.exists()
p.write_text(json.dumps({'schema':'TWO_STAGE_PIPELINE_SOURCE_AUTHORITY_V1','created_at':time.time(),'files':files,'physical_gpu_allowlist':[4,5],'historical_gpu23_evidence_preserved':True,'scientific_math_modified':False,'CPU_contract_tests_sha256':sha(A/'CPU_CONTRACT_TESTS.json'),'real_debug_artifact_cpu_audit_sha256':sha(A/'REAL_DEBUG_ARTIFACT_CPU_AUDIT.json'),'formal_training_started_at_source_freeze':False,'default_launch_mode':'dry-run','explicit_execute_required':True,'downstream_auto_start':False},indent=2,sort_keys=True)+'\n');p.with_suffix('.sha256').write_text(sha(p)+'  '+p.name+'\n')
print(json.dumps({'source_authority':str(p),'sha256':sha(p),'files':len(files)},indent=2))
