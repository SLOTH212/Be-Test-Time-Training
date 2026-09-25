import json,time
from pathlib import Path
R=Path('/path/to/ttt');A=R/'audits/v3';D=A/'analysis'
read=lambda name:json.loads((A/name).read_text())
assert read('GPU_PAUSE_USER_INSTRUCTION.json')['gpu_work_authorized_now'] is False
for name in ['RESUME_EQUIVALENCE_INDEPENDENT_AUDIT.json','REPORTING_INDEPENDENT_AUDIT.json','CPU_PARALLEL_INDEPENDENT_AUDIT.json']:assert read(name)['status']=='PASS'
stat=read('STATISTICAL_RESULT_V3.json');base=read('UNINTERRUPTED_NONDETERMINISM_BASELINE.json');res=read('RESUMED_NUMERICAL_VARIABILITY.json');assert stat['localization_required']
fmt=lambda src:'；'.join(g+' max_abs 的 median/p95/max='+ '/'.join(format(src[g]['distribution']['max_abs'][q],'.8g') for q in ['median','p95','max']) for g in ['model','optimizer'])
confirmed=len(stat['confirmed_abnormal']);triggers=len(stat['localization_required']);primary=stat['primary76_pass_count']
reason=('完整 CPU 分析按冻结协议得到 '+stat['statistical_case']+'：原始 76 项中 '+str(primary)+' 项通过全部经验门槛，'+str(confirmed)+' 项全状态/损失指标确认异常，'+str(triggers)+' 项触发条件定位。用户随后明确暂停 GPU，因此首个恢复更新的梯度/裁剪/优化器定位尚未执行。CPU 证据完整、独立审计通过，但 V3 最终结案和正式启动仍被阻断。')
answers=[
'精确模型参数为 model.layers.0.self_attn.o_proj.weight、model.layers.7.self_attn.q_proj.weight；74 个优化器状态的完整逐项名称、所属参数、形状和数值结果列在本报告“原始 76 项失败的精确清单及主结果”表中，来自 V2 原始比较文件。',
'没有集中到少数底层参数。74 个优化器失败项对应 71 个不同参数；其中 exp_avg=70、exp_avg_sq=4、step=0。加上 2 个模型失败项，合计关联 72 个参数。',
'未发现新的训练恢复状态遗漏。5 次新进程恢复的原始状态、输入和首个前向前 RNG 一致性通过。修复过一个诊断观察器读取尚未初始化 FSDP 字段的错误；它不是训练状态修复。',
fmt(base['group_aggregates'])+'。这些是完整运行对的描述性分布，不把运行对当作独立样本。',
fmt(res['RU'])+'；每张量 5 种指标、置信区间和方向结果均保留在机器审计中。',
('冻结统计结果为 '+stat['statistical_case']+'，确认异常项数='+str(confirmed)+'，触发定向定位项数='+str(triggers)+'。未通过非劣性置信区间本身不等同于确认异常；但极值超限也会阻止放行并要求定位。'),
'CPU 方向性检验状态为 '+stat['RESUME_DIRECTIONAL_DRIFT_STATUS']+'。完整标签置换及多重比较校正的逐项证据已保留；不将“不显著”解释为等价证明。',
'V2 原始数值失败明确保留。V3 新两两比较旧 allclose 的全状态结果为 '+('PASS' if all(stat['counts'][g]['legacy_allclose']==stat['counts'][g]['total'] for g in ['model','optimizer']) else 'FAIL')+'，未降低旧阈值或覆盖 V2 文件。',
'本次尚未获得 V3 放行结论，不能把 V2 失败改成非阻断警告。只有结构、输入、全部预注册经验门槛、方向性和独立审计均满足，且必需定位已闭合时，才能用新的独立证据说明自然变异，同时继续保留旧 FAIL。',
'是。确定性 A4 仅作为既有诊断证据；本轮 5+5 主实验保持 ORIGINAL，formal deterministic mode=false。',
'是。所有已执行 GPU 阶段固定物理 GPU2/3、WORLD_SIZE=2；没有使用其他 GPU。用户暂停后未启动条件 GPU 定位。',
'不能启动。SAFE_TO_LAUNCH=NO；完整 CPU 结果未闭合数值阻断，按协议要求的 GPU 定位因用户暂停而待办。没有创建 V3 放行权威或启动守卫。',
'是。正式 1B 训练、全量数据消费和 Stage2 均未启动。此 CPU 文档明确属于阶段性交付，未宣称 V3 目标全部完成。'
]
d={'created_at':time.time(),'required_work_complete':False,'SAFE_TO_LAUNCH':'NO','status':'INCONCLUSIVE','classification':'INCONCLUSIVE','numerical_equivalence':'FAIL' if stat['statistical_case']=='CASE_C' else 'INCONCLUSIVE','cpu_statistical_case':stat['statistical_case'],'reason':reason,'conditional_localization_completed':False,'localization_summary':'已完成异常触发检查，并准备选定参数的 pre-forward、pre-reduction、pre-clip、post-clip、post-step 只读观察代码；用户暂停 GPU 后未执行。阶段 21/22 的 GPU 证据缺失，不能声称已定位根因，也没有据此修改训练运行时。','remaining_resume_state_bug':False,'resume_bug_found':False,'blocking_phase':'CONDITIONAL_GRADIENT_LOCALIZATION_DEFERRED_BY_USER','blocking_reason':'Frozen empirical gates trigger conditional localization; user instruction temporarily prohibits GPU execution. CPU evidence alone does not complete this mandatory phase.','answers_13':answers}
(A/'CPU_REVIEW_DECISION.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print('CPU_REVIEW_DECISION_READY',stat['statistical_case'])
