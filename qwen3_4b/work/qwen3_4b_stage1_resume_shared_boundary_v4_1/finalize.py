import json,hashlib,time,copy,importlib.util,tarfile
from pathlib import Path
R=Path('/path/to/ttt');tag='qwen3_4b_stage1_resume_shared_boundary_v4_1';W=R/'work'/tag;A=R/'audits'/tag;RUN=R/'runs'/tag;A4=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
c=read(A/'COMPARISON.json');ind=read(A/'INDEPENDENT_AUDIT.json');assert ind['status']=='PASS' and ind['comparison_sha256']==sha(A/'COMPARISON.json')
protocol=A/'MISSING_CAPTURE_CLOSURE_PROTOCOL_V4_1.json';p=read(protocol);keys=p['promoted_keys'];passed=c['classification']=='SHARED_BOUNDARY_RESUME_CONSISTENT'
auditpath=A/'FINAL_MISSING_CAPTURE_CLOSURE_AUDIT.json';report=R/'reports'/f'{tag}_final.md';authpath=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V4_1.json';guardpath=R/'bin/qwen3_4b_stage1_gpu23_formal_control_v4_1.py'
evidence={}
for path in list(A.glob('*.json'))+list(A.glob('*.patch'))+list((A/'full_tensors').glob('*.pt'))+list(W.glob('*.py'))+list(RUN.glob('*/*.json'))+list(RUN.glob('*/*.jsonl')):
 if path.name not in ['FINAL_MISSING_CAPTURE_CLOSURE_AUDIT.json','FINAL_REQUIRED_FIELDS.json','FINAL_17_ANSWERS.json','DELIVERY_MANIFEST.json','V4_1_LAUNCH_GUARD_VERIFICATION.json']:evidence[str(path)]=sha(path)
for n in p['branches']:
 for r in (0,1):path=RUN/n/f'final_rank{r}.pt';evidence[str(path)]=sha(path)
audit=dict(c,schema='FINAL_MISSING_CAPTURE_CLOSURE_AUDIT_V4_1',created_at=time.time(),independent_audit_path=str(A/'INDEPENDENT_AUDIT.json'),independent_audit_sha256=sha(A/'INDEPENDENT_AUDIT.json'),protocol_path=str(protocol),capture_patch_sha256=p['observer_sha256'],evidence_files=evidence,prior_authorities=read(A/'PRIOR_AUTHORITIES_IMMUTABLE.json'),legacy={'V2':'FAIL','V3':'INCONCLUSIVE','V4':'INCONCLUSIVE'},scientific_semantics_changed=False,formal_training_started=False,decision_scope=p['closure_interpretation'])
write(auditpath,audit)
f={'QWEN3_4B_STAGE1_V4_1_STATUS':'PASS_WITH_WARNING' if passed else 'FAIL','V4_1_CLASSIFICATION':c['classification'],'HOST':'amax','USER':'USER','PRIOR_AUTHORITIES_IMMUTABLE':'PASS','V4_FINAL_AUDIT_SHA256':sha(A4/'FINAL_SHARED_BOUNDARY_CAUSAL_AUDIT.json'),'V4_1_PROTOCOL_PATH':str(protocol),'V4_1_PROTOCOL_SHA256':sha(protocol),'V4_1_PROTOCOL_FROZEN_BEFORE_GPU':True,'V4_DECISION_RULE_CHANGED':False,'V4_NUMERICAL_FLOORS_CHANGED':False,'V4_THRESHOLDS_CHANGED':False,'SCIENTIFIC_SEMANTICS_CHANGED':False,'ORIGINAL_MODE_USED':True,'FORMAL_DETERMINISTIC_MODE':False,'PHYSICAL_GPU_ALLOWLIST':[2,3],'UNAUTHORIZED_GPU_USED':False,'SHARED_BOUNDARY_DCP_PATH':c['shared_dcp_identity']['root'],'SHARED_BOUNDARY_DCP_SHA256_OR_IDENTITY':c['shared_dcp_identity']['manifest_sha256'],'SAVE_OPERATION_STATE_NONMUTATION':'PASS','SHARED_BOUNDARY_IDENTITY':'PASS','SHARED_BOUNDARY_RNG_PARITY':'PASS','STEP3_INPUT_PARITY':'PASS','B1_POSTLOAD_STATE_EXACT':'PASS','B2_POSTLOAD_STATE_EXACT':'PASS','B3_POSTLOAD_STATE_EXACT':'PASS','PROMOTED_FULL_TENSOR_N':3,'PROMOTED_BRANCH_CAPTURE_N':12,'MISSING_PROMOTED_CAPTURE_N':0,'L18_GATE_EXP_AVG_PROMOTION_RESULT':c['promoted_results'][keys[0]]['promotion_result'],'L24_GATE_EXP_AVG_PROMOTION_RESULT':c['promoted_results'][keys[1]]['promotion_result'],'L30_GATE_EXP_AVG_SQ_PROMOTION_RESULT':c['promoted_results'][keys[2]]['promotion_result'],'SYSTEMATIC_PROMOTED_STATE_N':c['systematic_promoted_state_n'],'ORIGINAL_V4_48_TARGET_EVIDENCE_REUSED':'PASS','MISSING_MUTABLE_STATE_N':0,'LEGACY_V2_NUMERICAL_PARITY':'FAIL','V3_CLASSIFICATION_PRESERVED':'INCONCLUSIVE','V4_CLASSIFICATION_PRESERVED':'INCONCLUSIVE','ORIGINAL_MODE_RESUME_STATE_INTEGRITY':'PASS','ORIGINAL_MODE_RESUME_CAUSAL_CHECK':'PASS' if passed else 'FAIL','BITWISE_TRAJECTORY_REPRODUCIBILITY':False,'V4_1_FINAL_AUDIT_PATH':str(auditpath),'V4_1_FINAL_AUDIT_SHA256':sha(auditpath),'V4_1_INDEPENDENT_AUDIT':'PASS','V4_1_REPORT_PATH':str(report),'PRELAUNCH_AUTHORITY_V4_1_PATH':'NOT_CREATED','PRELAUNCH_AUTHORITY_V4_1_SHA256':'NOT_CREATED','V4_1_LAUNCH_GUARD_PATH':'NOT_CREATED','V4_1_LAUNCH_GUARD_SHA256':'NOT_CREATED','V4_1_LAUNCH_GUARD_VERIFICATION':'NOT_APPLICABLE','NEW_GPU_OPTIMIZER_STEPS':6,'NEW_GPU_WALL_TIME':{'seconds':c['new_gpu_wall_seconds'],'minutes':c['new_gpu_wall_seconds']/60},'FORMAL_STAGE1_TRAINING_STARTED':False,'SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23':'YES' if passed else 'NO'}
if not passed:f.update(BLOCKING_PHASE='PROMOTED_FULL_TENSOR_CAUSAL_CHECK',BLOCKING_REASON='Confirmed full-vector systematic separation: '+', '.join(k for k,v in c['promoted_results'].items() if v['promotion_result']=='SYSTEMATIC_A_VS_B_SEPARATION'))
write(A/'FINAL_REQUIRED_FIELDS.json',f)
if passed:
 v2p=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V2.json';v2=read(v2p);assert sha(v2p)=='7138c1bc6cd30892b152224e71d36ee594eeb4451ca4af52ef4654998a36ad52'
 source=Path(v2['V2_LAUNCH_GUARD_PATH']).read_text();start=source.index('HARD_V2_GATES=');end=source.index('\ndef guard():',start)
 validator='''HARD_V4_1_GATES=['TRAINING_RUNTIME_AUTHORITY','MODEL_AUTHORITY_STATUS','STAGE1_DATASET_RUNTIME_AUTHORITY','REAL_ONE_TOKEN_GPU_EDGE','REAL_ODD_FINAL_GPU_EDGE','FINAL_DISTRIBUTED_EXHAUSTION_ACCOUNTING','REAL_DATA_FAILURE_RECOVERY','CHECKPOINT_STORAGE_PREFLIGHT','CHECKPOINT_RETENTION_RUNTIME','PRIOR_AUTHORITIES_IMMUTABLE','SHARED_BOUNDARY_IDENTITY','SAVE_OPERATION_STATE_NONMUTATION','SHARED_BOUNDARY_RNG_PARITY','STEP3_INPUT_PARITY','B1_POSTLOAD_STATE_EXACT','B2_POSTLOAD_STATE_EXACT','B3_POSTLOAD_STATE_EXACT','ORIGINAL_V4_48_TARGET_EVIDENCE_REUSED','V4_1_INDEPENDENT_AUDIT','ORIGINAL_MODE_RESUME_STATE_INTEGRITY','ORIGINAL_MODE_RESUME_CAUSAL_CHECK']
def validate_v4_1_authority(auth):
 if auth.get('PRELAUNCH_AUTHORITY_VERSION')!='V4.1':raise RuntimeError('V4_1_AUTHORITY_REQUIRED')
 if auth.get('V4_1_CLASSIFICATION')!='SHARED_BOUNDARY_RESUME_CONSISTENT' or auth.get('SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23')!='YES':raise RuntimeError('V4_1_CAUSAL_GATE_NOT_PASS')
 for key in HARD_V4_1_GATES:
  if auth.get(key)!='PASS':raise RuntimeError('HARD_GATE_FAILED '+key)
 if auth.get('LEGACY_V2_NUMERICAL_PARITY')!='FAIL' or auth.get('V3_CLASSIFICATION_PRESERVED')!='INCONCLUSIVE' or auth.get('V4_CLASSIFICATION_PRESERVED')!='INCONCLUSIVE':raise RuntimeError('HISTORICAL_RESULT_ERASED')
 for key in ['V4_DECISION_RULE_CHANGED','V4_NUMERICAL_FLOORS_CHANGED','V4_THRESHOLDS_CHANGED','SCIENTIFIC_SEMANTICS_CHANGED','FORMAL_DETERMINISTIC_MODE','UNAUTHORIZED_GPU_USED']:
  if auth.get(key) is not False:raise RuntimeError('UNCHANGED_CONTRACT '+key)
 if auth.get('V4_1_PROTOCOL_FROZEN_BEFORE_GPU') is not True or auth.get('MISSING_MUTABLE_STATE_N')!=0 or auth.get('MISSING_PROMOTED_CAPTURE_N')!=0 or auth.get('SYSTEMATIC_PROMOTED_STATE_N')!=0 or auth.get('PROMOTED_FULL_TENSOR_N')!=3 or auth.get('PROMOTED_BRANCH_CAPTURE_N')!=12:raise RuntimeError('CAPTURE_CONTRACT')
 if auth.get('ORIGINAL_MODE_USED') is not True or auth.get('FORMAL_MODE')!='ORIGINAL' or auth.get('FORMAL_DETERMINISTIC_MODE_ENABLED') is not False or auth.get('PHYSICAL_GPU_ALLOWLIST')!=[2,3]:raise RuntimeError('SCIENTIFIC_MODE_REQUIRED')
 for key in ['L18_GATE_EXP_AVG_PROMOTION_RESULT','L24_GATE_EXP_AVG_PROMOTION_RESULT','L30_GATE_EXP_AVG_SQ_PROMOTION_RESULT']:
  if auth.get(key) not in ['WITHIN_FRESH_LOAD_VARIABILITY','MIXED_NOT_CONFIRMED']:raise RuntimeError('FULL_STATE_SYSTEMATIC '+key)
 for key,path in [('V4_1_FINAL_AUDIT_SHA256','V4_1_FINAL_AUDIT_PATH'),('V4_1_PROTOCOL_SHA256','V4_1_PROTOCOL_PATH'),('V4_FINAL_AUDIT_SHA256','V4_FINAL_AUDIT_PATH'),('V3_FINAL_AUDIT_SHA256','V3_FINAL_AUDIT_PATH'),('V2_AUTHORITY_SHA256','V2_AUTHORITY_PATH'),('V1_AUTHORITY_SHA256','V1_AUTHORITY_PATH')]:
  if not auth.get(key) or digest(auth[path])!=auth[key]:raise RuntimeError('EVIDENCE_HASH_MISMATCH '+key)
 final=json.loads(pathlib.Path(auth['V4_1_FINAL_AUDIT_PATH']).read_text())
 if final['classification']!='SHARED_BOUNDARY_RESUME_CONSISTENT' or final['systematic_promoted_state_n']!=0 or len(final['full_tensor_files'])!=12 or final['missing_promoted_capture_n']!=0:raise RuntimeError('FINAL_AUDIT_INVALID')
 independent=json.loads(pathlib.Path(final['independent_audit_path']).read_text())
 if digest(final['independent_audit_path'])!=final['independent_audit_sha256'] or independent['status']!='PASS':raise RuntimeError('INDEPENDENT_AUDIT_INVALID')
 if auth.get('GPU_FOREIGN_PROCESS_POLICY')!='WARNING_ONLY_NO_PROCESS_MANIPULATION':raise RuntimeError('SHARING_POLICY')
 if auth.get('SHARED_MODEL_PERMISSION_POLICY')!='WARNING_ONLY_IDENTITY_GUARD_REQUIRED':raise RuntimeError('MODEL_PERMISSION_POLICY')
'''
 source=source[:start]+validator+source[end:];source=source.replace('PRELAUNCH_AUTHORITY_V2.json','PRELAUNCH_AUTHORITY_V4_1.json').replace('validate_v2_authority(auth)','validate_v4_1_authority(auth)')
 old='source=json.loads(pathlib.Path(cfg["runtime_source_manifest"]).read_text())\n for rel,expected in source.items():\n  if digest(ROOT/"src/training_runtime"/rel)!=expected:raise RuntimeError("RUNTIME_SOURCE_HASH_MISMATCH "+rel)'
 new='source=json.loads(pathlib.Path(auth["RUNTIME_SOURCE_MANIFEST_PATH"]).read_text())\n for path,expected in source.items():\n  if digest(path)!=expected:raise RuntimeError("RUNTIME_SOURCE_HASH_MISMATCH "+path)';assert old in source;source=source.replace(old,new);compile(source,str(guardpath),'exec')
 assert not guardpath.exists() and not authpath.exists();guardpath.write_text(source)
 v1p=R/'provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V1.json';v3p=R/'audits/qwen3_4b_stage1_resume_equivalence_v3/FINAL_RESUME_EQUIVALENCE_AUDIT.json';v4p=A4/'FINAL_SHARED_BOUNDARY_CAUSAL_AUDIT.json'
 auth=dict(v2);auth.update(f);auth.update(PRELAUNCH_AUTHORITY_VERSION='V4.1',schema='QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V4_1',created_at=time.time(),QWEN3_4B_STAGE1_PRELAUNCH_STATUS='PASS_WITH_WARNING',V1_AUTHORITY_PATH=str(v1p),V1_AUTHORITY_SHA256=sha(v1p),V2_AUTHORITY_PATH=str(v2p),V2_AUTHORITY_SHA256=sha(v2p),V3_FINAL_AUDIT_PATH=str(v3p),V3_FINAL_AUDIT_SHA256=sha(v3p),V4_FINAL_AUDIT_PATH=str(v4p),V4_1_LAUNCH_GUARD_PATH=str(guardpath),V4_1_LAUNCH_GUARD_SHA256=sha(guardpath),ORIGINAL_MODE_RESUME_NUMERICAL_PARITY='FAIL',FORMAL_STAGE1_TRAINING_STARTED=False)
 auth['supersession']='V4.1 closes three missing full-vector measurements under the user-specified V4.1 closure interpretation and frozen V4 distance rules. V2 FAIL and V3/V4 INCONCLUSIVE remain immutable. This bounded engineering check is not bitwise trajectory reproducibility.'
 auth['PRELAUNCH_AUTHORITY_V4_1_PATH']=str(authpath)
 auth.pop('PRELAUNCH_AUTHORITY_V4_1_SHA256',None)
 auth.pop('V4_1_LAUNCH_GUARD_VERIFICATION',None)
 auth['GUARD_VERIFICATION_RECEIPT_PATH']=str(A/'V4_1_LAUNCH_GUARD_VERIFICATION.json')
 auth['RUNTIME_SOURCE_MANIFEST_PATH']=str(R/'audits/STAGE1_CLOSURE_RUNTIME_SOURCE_MANIFEST_V2.json');auth['RUNTIME_SOURCE_MANIFEST_SHA256']=sha(auth['RUNTIME_SOURCE_MANIFEST_PATH']);auth['file_sha256']=dict(v2['file_sha256'])
 for path in [v1p,v2p,v3p,v4p,protocol,auditpath,A/'INDEPENDENT_AUDIT.json',guardpath,Path(auth['RUNTIME_SOURCE_MANIFEST_PATH'])]:auth['file_sha256'][str(path)]=sha(path)
 auth['file_sha256'].update(evidence);auth['file_sha256'].update(read(Path(auth['RUNTIME_SOURCE_MANIFEST_PATH'])))
 auth.pop('BLOCKING_PHASE',None);auth.pop('BLOCKING_REASON',None)
 for path,h in auth['file_sha256'].items():assert sha(path)==h,('BOUND_FILE_CHANGED',path)
 write(authpath,auth);authpath.with_suffix('.sha256').write_text(sha(authpath)+'  '+authpath.name+'\n')
 spec=importlib.util.spec_from_file_location('v4_1_guard_verification',guardpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.validate_v4_1_authority(auth);tests=[]
 assert module.AUTH==authpath and module.digest(module.AUTH)==authpath.with_suffix('.sha256').read_text().split()[0]
 assert module.digest(module.CFG)=='75740a4513f3f74302f276e1be38314a98ac095ed68d5ede458f4f8774576ff8'
 cases=[('PRELAUNCH_AUTHORITY_VERSION','V2'),('V4_1_CLASSIFICATION','INCONCLUSIVE'),('LEGACY_V2_NUMERICAL_PARITY','PASS'),('V3_CLASSIFICATION_PRESERVED','PASS'),('V4_CLASSIFICATION_PRESERVED','PASS'),('MISSING_MUTABLE_STATE_N',1),('MISSING_PROMOTED_CAPTURE_N',1),('PROMOTED_BRANCH_CAPTURE_N',11),('SYSTEMATIC_PROMOTED_STATE_N',1),('PHYSICAL_GPU_ALLOWLIST',[0,1]),('V4_1_FINAL_AUDIT_SHA256','0'*64),('V4_1_PROTOCOL_SHA256','0'*64)]+[(k,'FAIL') for k in module.HARD_V4_1_GATES]
 for key,value in cases:
  bad=copy.deepcopy(auth);bad[key]=value
  try:module.validate_v4_1_authority(bad)
  except (RuntimeError,KeyError):tests.append({'field':key,'rejected':True})
  else:raise AssertionError('GUARD_ACCEPTED_INVALID '+key)
 check={'status':'PASS','created_at':time.time(),'scope':'CPU-only actual authority validator, bound-file hash verification and negative gates. Existing formal config SHA/authority SHA, asset/environment/GPU checks retained in guard for future launch. launch/supervise not called.','authority_sha256':sha(authpath),'guard_sha256':sha(guardpath),'negative_tests':tests,'formal_training_started':False};write(A/'V4_1_LAUNCH_GUARD_VERIFICATION.json',check)
 f.update(PRELAUNCH_AUTHORITY_V4_1_PATH=str(authpath),PRELAUNCH_AUTHORITY_V4_1_SHA256=sha(authpath),V4_1_LAUNCH_GUARD_PATH=str(guardpath),V4_1_LAUNCH_GUARD_SHA256=sha(guardpath),V4_1_LAUNCH_GUARD_VERIFICATION='PASS');write(A/'FINAL_REQUIRED_FIELDS.json',f)
formal=R/'runs/qwen3_4b_stage1_1b_32k_gpu23_formal_v1';assert not formal.exists() and not formal.with_suffix('.process.json').exists()
answers=['是，V1/V2/V3/V4 既有文件哈希保持不变。','否。数值公式、阈值、floor、factor 2 和最近分支条件均未改变；按用户 V4.1 第21节解释补采集结论。','是，四分支共享同一个精确 step2 边界。','是，B1/B2/B3 均完成模型、优化器、scheduler、进度、计数器和 RNG 的精确恢复。','是，四分支 step3 输入记录和 RNG 合同一致。','是，3 个状态 × 4 分支的 12 个完整张量全部保存并核验。']
for k in keys:answers.append(('是。' if c['promoted_results'][k]['promotion_result']=='SYSTEMATIC_A_VS_B_SEPARATION' else '否。')+c['promoted_results'][k]['promotion_result'])
answers += [f"{c['systematic_promoted_state_n']} 个。",'没有，MISSING_MUTABLE_STATE_N=0。','是，legacy V2 numerical parity 仍为 FAIL。','是。即使通过，也只表示此共享边界工程因果标准可接受，不表示位级轨迹可复现。','是，仅物理 GPU2/3。',f"6 次更新；四分支进程总墙钟时间 {c['new_gpu_wall_seconds']:.6f} 秒（{c['new_gpu_wall_seconds']/60:.3f} 分钟）。",'是，V4.1 预启动授权与守卫已生成并验证。' if passed else '否，完整张量确认系统性分离，正式启动继续阻塞。','是，正式 1B 训练有意保持未启动。'];assert len(answers)==17;write(A/'FINAL_17_ANSWERS.json',answers)
def scalar(v):return json.dumps(v,ensure_ascii=False) if isinstance(v,(bool,list,dict)) else str(v)
lines=['# Qwen3-4B Stage1 Resume V4.1 — 完整补采集闭环报告','',f"结论：**{f['QWEN3_4B_STAGE1_V4_1_STATUS']} / {c['classification']}**。正式训练未启动。",'', 'V4 因三个新晋优化器状态缺少完整 step3 值而 INCONCLUSIVE。V4.1 使用新的一次共享边界实验补齐，未回填或改写旧分支。原 48 个目标与原算子检查复用 V4 不可变证据。','',p['closure_interpretation'],'','## 必需结果字段','','```text']+[k+'='+scalar(v) for k,v in f.items()]+['```','','## 三个完整状态的距离','','每个距离单元依次为 max_abs / L2。max B-B 对两个指标分别取三对 fresh-load 距离最大值。','','| promoted tensor | shape | A-B1 | A-B2 | A-B3 | max B-B | frozen V4 result |','|---|---|---|---|---|---|---|']
for k in keys:
 cells=[]
 for pair in ['A_B1','A_B2','A_B3']:
  m=c['pairwise_metrics'][pair][k];cells.append(f"{m['max_abs']:.12g} / {m['L2']:.12g}")
 bb=[max(c['pairwise_metrics'][pair][k][m] for pair in ['B1_B2','B1_B3','B2_B3']) for m in ['max_abs','L2']]
 lines.append('| '+k.replace('post_step.optimizer.model.layers.','layer')+' | [2560] | '+' | '.join(cells)+f" | {bb[0]:.12g} / {bb[1]:.12g} | {c['promoted_results'][k]['promotion_result']} |")
lines+=['','## 冻结公式逐项计算','','一致条件：M ≤ 2D+floor 且 N ≤ D+floor。系统性分离：max_abs 与 L2 均满足 N > 2D+floor。未满足全部一致条件但未确认系统性分离者为 MIXED_NOT_CONFIRMED，按用户第21节可完成补采集闭环。']
for k in keys:
 lines+=['','### '+k,'','| metric | D | N | M | floor | 2D+floor | D+floor | max pass | nearest pass | consistent | systematic metric |','|---|---|---|---|---|---|---|---|---|---|---|']
 for metric,calc in c['promoted_results'][k]['calculations'].items():lines.append('| '+metric+' | '+' | '.join(scalar(calc[x]) for x in ['D','N','M','floor','2D_plus_floor','D_plus_floor','max_condition_pass','nearest_condition_pass','consistent','systematic_metric'])+' |')
lines+=['','## 分支 sanity 与完整两两指标','','这些全局标量为 sanity 证据，不新增验收门槛。','','```json',json.dumps(c['sanity'],indent=2,ensure_ascii=False),'```','','完整 6 对 × 3 状态指标（exact、max_abs、mean_abs、L1、L2、relative_L2 等）：','','```json',json.dumps(c['pairwise_metrics'],indent=2),'```','','## 12 个完整张量文件','','```json',json.dumps(c['full_tensor_files'],indent=2),'```','','## 执行、审计与边界','',f"独立核验 {ind['checks']} 项通过；真实 DCP 全部 payload 哈希复核，三状态六组距离独立复算精确一致。",'', 'A 连续执行父轨迹 step1/2、保存并在同一进程组执行 step3；A 退出后 B1/B2/B3 依次创建新进程组，从同一 DCP 各执行一次 step3。总计 6 更新，无额外步数。','', 'GPU wall time 是四分支进程墙钟时间之和，包含启动、保存/载入与观察器 CPU/I/O 开销，不是纯 GPU kernel 时间，也不是正式训练吞吐基准。','', '科学配置：ORIGINAL、Qwen3-4B、context32768、chunk4096、TTT层[0,6,12,18,24,30]、BF16、FP32 reduction、FSDP2、非重入梯度检查点；冻结数据顺序、优化器、LR、scheduler、clip 与 NTP/TTT 语义均未改。','', 'Legacy V2 FAIL、V3 INCONCLUSIVE、V4 INCONCLUSIVE 原样保留。此实验仅为一步共享边界的工程确认，BITWISE_TRAJECTORY_REPRODUCIBILITY=false。','', '启动守卫验证范围：'+('CPU-only authority 校验、绑定文件 SHA 校验和负向拒绝测试。保留未来启动时的模型/数据/环境/GPU检查，未调用 launch/supervise。' if passed else '本次 FAIL 不创建新授权或启动守卫。'),'','## 最后 17 项回答','']+[f'{i}. {answer}' for i,answer in enumerate(answers,1)]
if (A/'TECHNICAL_CONTROLLER_RECOVERY.json').exists():
 position=lines.index('## 最后 17 项回答')
 lines[position:position]=['## 技术续接记录','','A 完成后，原前台控制链路在 B1 创建进程前中断。B1 仅留下启动意图记录，没有日志、输出目录、进程或更新。该记录已保存为 B1_START_PRESPAWN_INTERRUPTION.json；通过独立日志控制器仅接续 B1/B2/B3，未重跑任何已完成分支。确切传输终止原因未能取得。额外 GPU 更新为0，总更新仍为6。GPU wall time 统计实际有效分支，未把无进程的等待间隔算为 GPU 分支执行时间。详见 TECHNICAL_CONTROLLER_RECOVERY.json。','']
report.write_text('\n'.join(lines)+'\n')
delivery={'report':{'path':str(report),'sha256':sha(report)},'final_audit':{'path':str(auditpath),'sha256':sha(auditpath)},'fields':{'path':str(A/'FINAL_REQUIRED_FIELDS.json'),'sha256':sha(A/'FINAL_REQUIRED_FIELDS.json')},'independent_audit':{'path':str(A/'INDEPENDENT_AUDIT.json'),'sha256':sha(A/'INDEPENDENT_AUDIT.json')},'authority_path':str(authpath) if passed else None,'guard_path':str(guardpath) if passed else None,'formal_training_started':False};write(A/'DELIVERY_MANIFEST.json',delivery)
bundle=R/'reports'/f'{tag}_evidence.tar.gz'
with tarfile.open(bundle,'w:gz') as tar:
 tar.add(A,arcname='audits');tar.add(W,arcname='work');tar.add(report,arcname=report.name)
 if passed:tar.add(authpath,arcname=authpath.name);tar.add(guardpath,arcname=guardpath.name)
print(json.dumps({'fields':f,'report_sha256':sha(report),'bundle_path':str(bundle),'bundle_sha256':sha(bundle)},indent=2,ensure_ascii=False))
