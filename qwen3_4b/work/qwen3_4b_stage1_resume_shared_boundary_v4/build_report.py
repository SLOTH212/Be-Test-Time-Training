import json,hashlib,time,tarfile,math
from pathlib import Path
R=Path('/path/to/ttt');W=R/'work/qwen3_4b_stage1_resume_shared_boundary_v4';A=R/'audits/qwen3_4b_stage1_resume_shared_boundary_v4';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=read(A/'FINAL_REQUIRED_FIELDS.json');audit=read(A/'FINAL_SHARED_BOUNDARY_CAUSAL_AUDIT.json');d=audit['decision'];passed=f['SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23']=='YES';names=d['branches'];agg=audit['aggregate_target_comparisons'];ind=read(A/'INDEPENDENT_AUDIT.json');assert ind['status']=='PASS'
if passed:assert f['V4_LAUNCH_GUARD_VERIFICATION']=='PASS'
else:
 f.update(PRELAUNCH_AUTHORITY_V4_PATH=None,PRELAUNCH_AUTHORITY_V4_SHA256=None,V4_LAUNCH_GUARD_PATH=None,V4_LAUNCH_GUARD_SHA256=None,V4_LAUNCH_GUARD_VERIFICATION='NOT_CREATED_NOT_AUTHORIZED_TO_LAUNCH');(A/'FINAL_REQUIRED_FIELDS.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
fmt=lambda x:format(x,'.9g')
def pair_summary(pair):
 return '；'.join(k+' max_abs='+fmt(agg[pair][k]['max_abs'])+'、L2='+fmt(agg[pair][k]['L2']) for k in ['MODEL','OPTIMIZER','GRADIENT','POSTCLIP'])
answers=[
'是。A、B1、B2'+('、B3' if 'B3' in names else '')+'都来自同一条父轨迹第2步的精确状态；没有独立重跑前两步。',
'没有。两卡B_PRE_SAVE与B_POST_SAVE的模型、优化器、scheduler、cursor、计数器、RNG和已审计训练状态精确一致。',
'是。B1/B2在真实DCP加载后及第3步前均与共享保存状态精确一致。',
'是。所有分支消费同一组canonical step3记录，record IDs、token计数和RNG一致。',
'B1–B2：'+pair_summary('B1_B2')+'。这里的汇总只覆盖47个目标关联参数及其矩状态。',
'A–B1：'+pair_summary('A_B1')+'。A–B2：'+pair_summary('A_B2')+'。',
('未发现超过预注册接受规则的A–B差异。' if passed else '存在未通过预注册接受规则的观察项，具体见逐项表；不能以整体平均掩盖局部差异。')+'目标计数='+str(f['TARGETS_WITH_A_B_EXCESS_OVER_B_B'])+'。',
'前向效应分类：'+f['PROCESS_RESTART_FORWARD_EFFECT']+'。这是本共享边界短实验的结果，不是所有CUDA路径的确定性证明。',
'是。每个分支全部47个关联参数的实际归约梯度，都逐元素符合两卡本地贡献的FP32平均并转换到实际梯度dtype。',
'是。每个分支的真实post-clip向量都符合其实际pre-clip输入和现场记录的裁剪系数。',
'完整目标向量的优化器效应分类：'+f['OPTIMIZER_RESTART_SPECIFIC_EFFECT']+'。所有目标step从2推进到3；另有最大值摘要候选，但3个新增非目标状态缺少完整终态，不能完成要求的提升比较。',
'未发现。MISSING_MUTABLE_STATE_N=0；本轮保存、加载、首次前向前检查均通过。',
'是。V2 legacy numerical parity保持FAIL，V3保持INCONCLUSIVE，原权威未改写。',
('本次在同一精确边界下支持恢复一致性，解决了V3独立父轨迹造成的归因混杂。' if passed else '未完全解决。V4消除了V3的独立父轨迹混杂，但新增非目标候选缺少完整向量；最终结论为'+f['V4_CLASSIFICATION']+'，不能因此放行。')+'结论范围是一处边界、一次后续更新；不是长程数学等价。',
'是。仅物理GPU2/3、WORLD_SIZE=2，未切换其他GPU或操纵其他用户进程。',
'新增'+str(f['NEW_GPU_OPTIMIZER_STEPS'])+'次优化器更新，GPU阶段墙钟合计'+fmt(f['NEW_GPU_WALL_TIME'])+'秒（'+fmt(f['NEW_GPU_WALL_TIME']/60)+'分钟）。含构建、真实DCP及观测导出，不能当吞吐基准。',
('是，依据V4共享边界因果证据和独立审计，可按V4权威及守卫启动；本任务不执行启动。' if passed else '否。SAFE_TO_LAUNCH=NO，未创建V4放行权威或启动守卫。'),
'是。正式1B及Stage2均未启动，本轮全部输出为DEBUG_ONLY。'
]
lines=['# Qwen3-4B Stage1 Resume：Shared-Boundary Causal Fork V4','','**最终状态：'+f['QWEN3_4B_STAGE1_SHARED_BOUNDARY_V4_STATUS']+'；分类：'+f['V4_CLASSIFICATION']+'；正式训练放行：'+f['SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23']+'。**','','## 必须保留的证据缺口','','完整目标向量和算子核验已经完成；摘要阈值计算产生PROCESS_RESTART_OR_RESUME_SPECIFIC_EFFECT候选，原始ABCD_DECISION未改写。最终审计发现其中3个新增非目标状态没有保存第3步完整向量，无法完成用户第18节要求的提升比较。按照GPU执行前冻结的missing-capture规则，最终分类为INCONCLUSIVE；这不是修改阈值，也不是因为V2 legacy allclose失败。','','```json',json.dumps(audit['measurement_coverage'],ensure_ascii=False,indent=2),'```','','## 实验回答的问题','','V3中U_LOC和R_LOC分别训练到第2步，保存前状态已经不同，所以不能从后续差异判断恢复的额外影响。V4只训练一条父轨迹，产生唯一的共享边界B：optimizer step=2、cursor=4、实际input tokens=115086。A保存后在同一个进程组继续；B1/B2各自新建进程组，加载这同一份DCP，执行完全相同的第3步。','','因此A–B比较同进程继续与新进程恢复；B–B比较精确相同恢复状态下的执行波动。共享边界DCP的manifest SHA256：`'+f['SHARED_BOUNDARY_DCP_IDENTITY']+'`。','','## 冻结协议与范围','','GPU启动前的协议SHA256：`'+f['V4_PROTOCOL_SHA256']+'`。未改变科学配置：ORIGINAL、BF16、FP32归约、FSDP2、非重入梯度检查点、32K context、4K chunk、两卡/每卡micro=1/GA=1、原优化器和数据顺序。','','预注册规则对每个观察量使用新加载分支的最大两两差异D、A到新加载分支的最小差异N和最大差异M。接受条件是M≤2D+floor且N≤D+floor；不会把所有张量平均后放行。因子2及最近邻约束沿用V3极值保护的形式。本次是新共享边界的工程判定，不重跑或改写V3置信区间。','','参数/矩状态floor沿用既有绝对分辨率1e-5/1e-6。梯度和裁剪明确预注册使用1e-6，前向采样使用1e-5；这些是工程分辨率，不冒称历史梯度自然波动分位数。L1/L2分别按n和sqrt(n)缩放floor。完整规则与每项实际floor保存在协议和机器审计。','','A/B1/B2若有观察项未通过，会且只会触发一次B3；B3仍加载同一DCP。B3后若max_abs和L2都显示A与所有新加载分支分离超过2D+floor，则报告进程重启/执行路径效应；如果仍无法区分，则报告INCONCLUSIVE。未在观察结果后更改规则。','','## 分支证据','','| Branch | Execution class | Parent boundary | Step3 records | Load mode | Valid |','|---|---|---|---|---|---|']
records=d['contracts'][0]['sample_ids']
for n in names:lines.append('| '+n+' | '+('same-process continuation' if n=='A' else 'fresh-process load')+' | SAME_BOUNDARY_B | '+', '.join(records)+' | '+('saved once; no reload' if n=='A' else 'same immutable A/checkpoints')+' | VALID |')
lines+=['','B3：'+('按ABC预注册判定触发，新增1次更新。' if 'B3' in names else '未触发，未执行。'),'','保存非变异、边界身份、RNG一致、B1/B2加载后精确一致均为PASS。各个新进程组在前一组正常退出后才启动。','','## 主要比较','','以下是47个目标关联唯一参数的汇总；优化器只汇总94个一/二阶矩张量，不包含step计数。它们不是全模型逐元素比较结果。其他状态采用结构、finite、norm、摘要及哈希检查。','','| Pair | Group | max_abs | mean_abs | L1 | L2 | relative L2（pair左侧参考） |','|---|---|---:|---:|---:|---:|---:|']
for pair,groups in agg.items():
 for k,v in groups.items():lines.append('| '+pair+' | '+k+' | '+' | '.join(fmt(v[z]) for z in ['max_abs','mean_abs','L1','L2','pair_left_reference_relative_L2'])+' |')
lines+=['','## 前向、归约、实际裁剪及优化器','','前向效应：`'+f['PROCESS_RESTART_FORWARD_EFFECT']+'`。优化器效应：`'+f['OPTIMIZER_RESTART_SPECIFIC_EFFECT']+'`。','','前向保存完整内容哈希、norm/max/mean、finite计数和固定33个采样位置。采样差异不等于完整激活的逐元素距离；完整激活哈希仅用于精确相等判断。函数式调用的model.layers.6.mlp.ttt_proj不触发独立模块hook，按预注册方案比较所属MLP输出，并完整捕获投影参数的梯度与矩状态。','','每分支47个参数的归约和裁剪分别独立重建并逐元素核验。归约使用两卡FP32平均再cast；实际裁剪核验使用现场实际系数与真实post-clip向量，未使用min(norm,threshold)代替测量。','','| Branch | step3 loss | pre-clip global norm | actual coefficient |','|---|---:|---:|---:|']
for n in names:
 m=read(R/'runs/qwen3_4b_stage1_resume_shared_boundary_v4'/n/'final_rank0.json');lines.append('| '+n+' | '+fmt(d['training_step3'][n]['global_mean_loss'])+' | '+fmt(m['clip_observation']['global_norm']['samples'][0])+' | '+fmt(m['clip_observation']['actual_clamped_coefficient']['samples'][0])+' |')
lines+=['','## 精确48项目标','','“超出”指目标关联观察项未满足预注册规则；“系统分离候选”指max_abs和L2同时超过所有新加载分支的2D+floor。完整每阶段数值和判定均在机器审计。','','| # | V3冻结目标 | 关联参数 | 超出观察项数 | 系统分离候选数 |','|---:|---|---|---:|---:|']
for i,t in enumerate(audit['target_comparisons'],1):lines.append('| '+str(i)+' | `'+t['target']+'` | `'+t['parameter']+'` | '+str(len(t['excess_observables']))+' | '+str(len(t['systematic_observables']))+' |')
lines+=['','## 审计、历史结论与限制','',f"独立审计PASS，共{ind['check_count']}项检查。它独立复算完整目标指标、compact观察量、阈值来源、判定规则及共享边界/RNG证据。审计PASS不自动等于训练放行；放行还要求分类为SHARED_BOUNDARY_RESUME_CONSISTENT。",'','V1/V2/V3权威与证据未修改；V2 legacy FAIL、V3 INCONCLUSIVE原样保留。没有重复5+5、旧bootstrap、全状态两两大矩阵、吞吐、边缘样本、故障恢复或Stage2测试。','','观察器只读，但CPU复制和落盘会改变同步时机；不声称无时间扰动。实验仅覆盖一个共享边界和一个后续更新，不能证明长程轨迹数学相同或CUDA确定性。','','## 完整要求字段','','```json',json.dumps(f,ensure_ascii=False,indent=2),'```','','## 18项回答','']
for i,answer in enumerate(answers,1):lines.extend([str(i)+'. '+answer,''])
report=Path(f['V4_REPORT_PATH']);report.write_text('\n'.join(lines));(A/'FINAL_18_ANSWERS.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n')
# Deliver compact evidence; raw tensor files and DCP remain server-side with bound identities.
files=[report]+list(A.glob('*.json'))+list(W.glob('*.py'))
if passed:files += [Path(f['PRELAUNCH_AUTHORITY_V4_PATH']),Path(f['V4_LAUNCH_GUARD_PATH'])]
manifest={str(p.relative_to(R)):sha(p) for p in files if p.name!='DELIVERY_MANIFEST.json'};mp=A/'DELIVERY_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');bundle=R/'reports/qwen3_4b_stage1_resume_shared_boundary_v4_evidence.tar.gz'
with tarfile.open(bundle,'w:gz') as tf:
 for p in files+[mp]:tf.add(p,arcname=str(p.relative_to(R)))
print(json.dumps({'report':str(report),'report_sha256':sha(report),'bundle':str(bundle),'bundle_sha256':sha(bundle),'audit_sha256':sha(A/'FINAL_SHARED_BOUNDARY_CAUSAL_AUDIT.json')},indent=2))
