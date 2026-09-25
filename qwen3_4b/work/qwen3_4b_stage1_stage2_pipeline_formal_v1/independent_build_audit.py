"""Independent read-only build verification, no trainer/launcher execution."""
import ast,copy,json,os,time
from pathlib import Path
from unittest.mock import patch
import pipeline_core as core
R=core.ROOT;W=R/'work/qwen3_4b_stage1_stage2_pipeline_formal_v1';A=R/'audits/qwen3_4b_stage1_stage2_pipeline_formal_v1';c=core.config();checks=[]
def check(value,label):
 if not value:raise AssertionError(label)
 checks.append(label)
source=core.source_contract();check(source['physical_gpu_allowlist']==c['physical_gpu_allowlist']==[4,5],'GPU45 future policy')
for path,h in core.read(A/'PRIOR_SOURCE_AUTHORITIES.json')['files'].items():check(core.digest(path)==h,'prior immutable '+path)
a1=core.read(c['stage1']['authority']);a2=core.read(c['stage2']['authority'])
check(core.digest(c['stage1']['authority'])=='344795e0c0775265e639923bbff41d28a5a8b78339b3e2d25cd2d4e6e3b04e29','V4.1 identity')
check(a1['STAGE1_PACKAGE_SHA256']=='11ddc46d95381c6bdaed7f3a62a7f2f2711e95db1223f7f50ec85e249832217d','Stage1 parent package identity')
check(a2['STAGE2_PACKAGE_SHA256']=='60a54b694982a4a1c21708ba7c177b25101a5bc76a771f542f90e745611ce230','Stage2 package identity')
check(a1['LEGACY_V2_NUMERICAL_PARITY']=='FAIL' and a1['V3_CLASSIFICATION_PRESERVED']==a1['V4_CLASSIFICATION_PRESERVED']=='INCONCLUSIVE','historical warning preservation')
from gpu45_resource_loader import adapt,execute_worker,resource_contract
resource=resource_contract();check(resource['scientific_semantics_changed'] is False,'resource-only authority')
for name in ['distributed_train_worker.py','distributed_stage2_train_worker_v1.py']:
 original=(R/'src/training_runtime/code/workers'/name).read_text();new=adapt(original)
 reverse=new.replace('"4,5"','"2,3"').replace('"physical_gpu": 4 + context.local_rank','"physical_gpu": 2 + context.local_rank')
 check(reverse==original,'all scientific bytes unchanged '+name);check(ast.dump(ast.parse(reverse))==ast.dump(ast.parse(original)),'AST equality '+name)
for visible in ['0,1','2,3','4','4,5,6','']:
 with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visible,'WORLD_SIZE':'2'}):
  try:execute_worker('/nonexistent_should_never_be_imported','__main__')
  except RuntimeError as e:check('GPU45_REQUIRED' in str(e),'pre-import GPU rejection '+visible)
  else:raise AssertionError('GPU_POLICY_BYPASSED')
tests=core.read(A/'CPU_CONTRACT_TESTS.json');check(tests['status']=='PASS' and len(tests['tests'])>=41,'41 CPU contract tests')
check(all(x['status']=='PASS' for x in tests['tests']),'all tests pass');check(tests['source_sha256']==core.digest(W/'test_contracts.py'),'test source identity')
real=core.read(A/'REAL_DEBUG_ARTIFACT_CPU_AUDIT.json');check(real['status']=='PASS' and real['stage1_proven_DCP_to_model_only_exact'] and real['inference_meta_architecture_compatible'],'real DEBUG CPU route and inference compatibility')
check(real['GPU_used'] is False and real['formal_final_created'] is False and real['no_forward_or_RULER_executed'],'no GPU or formal fixture claim')
check(real['stage1_keys']==real['stage2_keys']==417 and real['unique_parameter_count']==4061881856,'state-dict inventory')
check(real['verification_source_sha256']==core.digest(W/'verify_real_debug_artifacts.py'),'real artifact verifier source')
text=(A/'DRY_RUN.log').read_text();needle='{\n  "mode": "DRY_RUN"';start=text.find(needle);check(start>=0,'default wrapper dry-run completed');dry=json.loads(text[start:]);check(dry['preflight']['status']=='PASS' and dry['formal_training_started'] is False,'actual default wrapper preflight')
check('"guard": "PASS"' in text,'actual Stage1 guard verify-only accepted')
for stage in (1,2):
 check(not Path(c[f'stage{stage}']['final_root']).exists(),'no fabricated FINAL '+str(stage))
check(not Path(c['stage1']['run_root']).exists() and not Path(c['stage2']['pointer']).exists(),'no formal training during build')
check(not (Path(c['state_root'])/'EXECUTION_AUTHORIZED.json').exists(),'no execute during build')
sup=(W/'pipeline_supervisor.py').read_text();prune=(W/'pipeline_prune.py').read_text();final=(W/'pipeline_finalize.py').read_text()
check('flock' in (W/'pipeline_core.py').read_text() and 'pass_fds=(self.fd,)' in sup,'exclusive lock includes launch window')
check("'BLOCK_UNCERTAIN_OR_FAILED_PRIOR_ATTEMPT'" in sup,'uncertain launch never retried')
check("'FINAL_MODEL_ARTIFACT_N':2" in sup and 'self.finish()' in sup,'exactly two final models')
check('pipeline_prune' not in sup and 'rmtree' not in sup,'no automatic checkpoint deletion')
check("args.execute" in prune and 'PIPELINE_NOT_COMPLETE' in prune,'explicit final-gated cleanup')
check('os.rename(temp,final)' in final and 'REFUSE_FINAL_OVERWRITE' in final,'atomic final publish and overwrite refusal')
check('source_contract();preflight(c)' in final,'finalizer verifies authority before publication')
check(c['downstream_auto_start'] is False and not any(x in sup for x in ['ruler','Fixed8','Dynamic','mechanism.py']),'stop after Stage2 FINAL')
result={'status':'PASS','checked_at':time.time(),'check_count':len(checks),'checks':checks,'GPU_used':False,'formal_training_started_during_build':False,'stage1_finalizer_ready':'PASS','stage2_finalizer_ready':'PASS','exact_parent_parser_contract':'PASS','exactly_once_semantics':'PASS','supervisor_recovery':'PASS','source_authority_sha256':core.digest(core.SOURCE),'config_sha256':core.digest(core.CONFIG),'dry_run_log_sha256':core.digest(A/'DRY_RUN.log'),'cpu_contract_tests_sha256':core.digest(A/'CPU_CONTRACT_TESTS.json'),'real_debug_artifact_audit_sha256':core.digest(A/'REAL_DEBUG_ARTIFACT_CPU_AUDIT.json'),'auditor_sha256':core.digest(Path(__file__)),'limitations':['GPU45 is an explicit resource-only adaptation of GPU23-validated scientific code; no new GPU trial in build.','Worker failure blocks pipeline; recovery script restarts supervisor only, never silently reruns a failed trainer.','Pre-spawn uncertainty fails closed instead of attempting an unsafe duplicate launch.']}
core.atomic(A/'INDEPENDENT_BUILD_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
