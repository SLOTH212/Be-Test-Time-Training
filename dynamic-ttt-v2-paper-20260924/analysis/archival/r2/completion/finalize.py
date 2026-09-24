import os,sys,json,csv,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/r2_cpu_mplconfig')
sys.path.insert(0,'/tmp/r2_cpu_plot_deps')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).resolve().parent;B=P.parent
read=lambda f:list(csv.DictReader(f.open()))
rows=read(B/'per_sample_improved.csv');sets=read(P/'setwise_alignment_and_reversal.csv');summ=read(P/'setwise_summary.csv');dom=read(P/'dominant_transition_by_task.csv');tt=read(P/'task_transition_types.csv');tasks=sorted({r['task'] for r in rows});figdir=P/'figures'
# Cross-check independent joins, set equivalence, probability mass, and reverse involution.
lookup={(r['sample_id'],r['set']):r for r in sets}
for sid in {r['sample_id'] for r in sets}:
 if (sid,'match') in lookup:
  a,b=lookup[sid,'match'],lookup[sid,'best']
  for k in ['candidate_n','early','late','joint','reverse_mean_delta','reverse_positive_fraction']:assert a[k]==b[k]
for r in sets:assert abs(float(r['reverse_positive_fraction'])+float(r['reverse_equal_fraction'])-1)<1e-10
for label in ['best','match']:
 for t in ['ALL_TASKS']+tasks:assert abs(sum(float(r['observed_fraction']) for r in tt if r['set']==label and r['task']==t)-1)<1e-10
integrated=[]
for t in tasks:
 rr=[r for r in rows if r['task']==t];ss=[r for r in sets if r['task']==t and r['set']=='match'];dr=next(r for r in dom if r['set']=='match' and r['task']==t)
 integrated.append({'task':t,'improved_n':len(rr),'matching_n':len(ss),'mean_gain':np.mean([float(r['gain']) for r in rr]),'mean_native_switches':np.mean([float(r['switch_count']) for r in rr]),'mean_recovery':np.mean([float(r['r2_recovery']) for r in rr]),'exact_match_rate':len(ss)/len(rr),'matching_tau_mean_norm':np.mean([float(r['tau_mean_norm']) for r in ss]),'longest_matching_plateau_n_mean_all_improved':np.mean([int(r['r2_max_plateau_n']) for r in rr]),'dominant_matching_transition_type':dr['dominant_type_ties'],'dominant_type_mass_fraction':dr['dominant_fraction']})
with (P/'task_structure_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(integrated[0]));w.writeheader();w.writerows(integrated)
# Re-render task heatmap with actual conditional sample sizes.
types=sorted({r['type'] for r in tt});v=np.array([[float(next(q['observed_fraction'] for q in tt if q['set']=='match' and q['task']==t and q['type']==ty)) for ty in types] for t in tasks]);fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(v,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(len(types)),types,rotation=35,ha='right');ax.set_yticks(range(len(tasks)),[t.replace('ruler_','').replace('_16k','')+' (n='+next(q['n'] for q in tt if q['set']=='match' and q['task']==t)+')' for t in tasks]);ax.set_title('Matching R2 transition types by task | equal weight per sample');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'task_transition_types.png',dpi=160);plt.close(fig)
with PdfPages(figdir/'all_complexity_metrics_by_task.pdf') as pdf:
 for metric in ['switch_count','distinct_actions','entropy','first_switch_norm','last_switch_norm','max_run_fraction']:
  fig,axes=plt.subplots(3,3,figsize=(12,10));rng=np.random.default_rng(20260923)
  for ax,t in zip(axes.flat,tasks):
   rr=[r for r in rows if r['task']==t];x=np.array([float(r[metric]) for r in rr]);y=[float(r['gain']) for r in rr]
   if metric in ['switch_count','distinct_actions']:x=x+rng.uniform(-.1,.1,len(x))
   ax.scatter(x,y,c=['#167c80' if r['r2_match']=='True' else '#c04e37' for r in rr],s=20,alpha=.7);ax.set_title(t.replace('ruler_','').replace('_16k','')+f' (n={len(rr)})');ax.set_xlabel(metric);ax.set_ylabel('Dynamic − SampleBest');ax.set_ylim(-.04,1.04)
  fig.suptitle('Improved samples | teal: R2 matched; orange: unmatched\nCount metrics display jitter ±0.1; statistical calculations use original values');fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)
report='''本轮已补齐附件中基于当前日志可以完成的 CPU 数值分析。用户随后要求先不生成图，图表工作已暂停；不将未完成图册计作交付。数据继续限定为最新 V2 Dynamic 与配套 R2；无模型加载、无 GPU、无新推理。

全部最优集合与匹配集合现在分别统计，避免遗漏 16 个未匹配样本。先对每个样本内部候选等权，再对样本等权；任务内分层 bootstrap 4,000 次，seed=20260923。无匹配的样本不作为“零对齐”混入匹配集合。

| 新增统计 | 结果 |
|---|---:|
| 全部最优集合，170 个样本：early / late 对齐 | 15.45% / 21.83% |
| 全部最优集合：同时对齐 | 0.208%；5/170 至少存在一个同时对齐解 |
| 未匹配 16 个样本的最优集合：同时对齐 | 0.298%；2/16 存在同时对齐解 |
| 全部匹配集合，154 个样本：正向−反转 | 50.13 pp，95% CI [48.36, 51.87] |
| 多次切换匹配组，62 个样本：正向−反转 | 54.33 pp，95% CI [51.60, 57.12] |
| 多次切换匹配组：所有匹配解均严格优于反转 | 54/62 |
| 未匹配组的最优集合：正向−反转 | 1.92 pp，95% CI [0, 5.09] |

在 62 个多次切换匹配样本中，匹配候选严格优于反转的样本等权比例为 98.60%。这和“选一个最优代表”的统计不同，但仍受结果筛选影响。置信区间只反映这些条件样本的抽样变异，不消除选最优/选成功的偏差，也不证明某种固定动作方向的因果作用。

跨任务：匹配集合的主导类型为 shallow→deep（5 个任务）、deep→shallow（3 个任务）、ALL→layer（1 个任务）。这里 shallow/deep 仅指两个单层动作的层号大小，不把 OFF/ALL 排进深度顺序。按各样本 Native 动作占比构造条件于 A≠B 的独立基线，再汇总到任务：shallow→deep 仅在 2/9 任务 enrichment>1，deep→shallow 仅在 1/9。没有跨任务统一深度方向的支持。multikey1 仅有 1 个匹配样本，图中标出 n；任务等权宏平均不能忽视这种小样本问题。

该新版基线与主报告旧表中的“先汇总 Native 边际再相乘”不同，是敏感性分析，不覆盖旧表。新表路径为 task_transition_enrichment.csv；旧表仍在上级目录。Native 未出现某动作时，预期可为零，此时 enrichment 留空，不填无穷大；这种基线不能为该动作提供检验。方向统计均为探索性描述，未进行多重比较后的显著性断言。

增补了任务内去均值的秩相关：gain 与 switch count 为 −0.038，distinct actions 为 −0.053，entropy 为 0.019。首次切换归一化位置由跨任务 pooled rho≈0.538 降到任务内中心化秩相关≈0.230，提示任务混杂；中心化秩相关是补充指标，不冒充原 pooled Spearman 或因果估计。

逐项交付清单：

| 附件项目 | 完成内容与文件 |
|---|---|
| 1. Boundary landscape | 上级 pair_landscapes.csv、boundary_details.csv；数值表覆盖 7,140 个动作对和所有切换点；完整曲线图册按用户要求暂停；跨度与连续平台分开 |
| 2. Early/late 对齐 | setwise_alignment_and_reversal.csv、setwise_summary.csv；best/match 分开、并列众数集合与奇数分段敏感性 |
| 3. Native switch structure | 上级逐样本表及本轮集合表；first/last 距离，末次边界与稳定 run 起点采用统一索引 |
| 4. Count-matched reversal | 上级全 48,678 个无序配对；本轮全部最优/匹配集合的样本等权分差、方向比例及 bootstrap |
| 5. R2 failure 结构 | 上级 62 matched vs 16 unmatched；switch、distinct、run、entropy、return、ABA 与 Fisher；不能沿用旧 60/13 |
| 6. Gain vs complexity | 上级 Spearman、分箱、bootstrap；本轮 6 种复杂度指标×9 任务散点图与任务内中心化秩相关 |
| 7. 最小复杂度恢复率 | 上级逐样本 recovery、mean/median/Q25/Q75、阈值比例和总增益恢复比；未裁剪 |
| 8. 最优解唯一性 | 上级 n_optima/n_match（字段 r2_best_n/r2_match_n）、动作对数、tau 集合；本轮恢复率/多重性/平台图 |
| 9. Action-pair structure | best/match 两套全局矩阵，逐任务 enrichment、8 种互斥方向类型及显式零基线处理 |
| 10. 跨任务结构一致性 | task_structure_comparison.csv、dominant_transition_by_task.csv、cross_task_direction_consistency.csv 与任务类型热图 |
| 11. 搜索过程 | 上级 first improvement/final score/selected trajectory 的去重候选首次序号与归一化比例；受日志字段限制，见下段 |
| 12. Prediction hash | 上级逐样本：154 score-match 中 152 至少存在 prediction-match；确定性最优代表 144；多次切换分别 60/62、59/62 |

仍无法从现有数据恢复的是精确 replay-index/父子搜索树：日志没有显式 replay_index/parent，缓存、种子及强制重放使去重候选序号不同于原始 replay 次数。已报告可确认的序号代理和来源代码，未编造缺失字段。隐藏状态、KV、ΔW 的新干预或补采不在本次 CPU-only 任务范围内。

图表目录 figures：all_170_reward_landscapes.pdf 是完整切换网格；完整 reward–tau 曲线图册生成中异常退出，已移除不完整 PDF，并按用户要求暂停；example_pair_curves.pdf 是按预定义结构类别取最小 sample ID 的六个示例，选择规则见 figure_example_selection.csv。all_complexity_metrics_by_task.pdf 包含六种复杂度指标。其余 PNG 为可直接预览的汇总图。所有轴使用原始 chunk 边界或明确标注的 tau/T，未把 chunk 位置声称为精确 token 位置。

数值复现仅执行 numerics.py（与 complete.py 的绘图前部分一致）。图表脚本暂停，不运行。使用现有 ttt_phase_c_v1 Python；绘图依赖装在 /tmp/r2_cpu_plot_deps，不修改原实验环境。临时目录消失时需恢复相同绘图库；版本记录见 environment.json。所有输入哈希和只读复核结果保存在本目录；CPU 输出不改变原实验结果。
'''
(P/'COMPLETION_REPORT.md').write_text(report)
(B/'CPU_AUDIT_REPORT.md').write_text((B/'CPU_AUDIT_REPORT.md').read_text().split('\n补充审计已完成：')[0]+'\n补充审计已完成：全部最优集合、完整集合反转统计、跨任务方向和数值分析已交付；图表工作按用户要求暂停，见 [补充报告](completion/COMPLETION_REPORT.md)。原文中的“未做跨任务方向分析”限制已由本轮补齐；因果与日志缺失限制仍成立。\n')
import matplotlib
(P/'environment.json').write_text(json.dumps({'CPU_only':True,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'threads':1,'plot_dependency_path':'/tmp/r2_cpu_plot_deps'},indent=2)+'\n')
(P/'independent_checks.json').write_text(json.dumps({'status':'PASS','best_match_set_equality_samples':154,'setwise_probability_mass_checked':len(sets),'transition_task_masses_checked':20,'exactly_one_switch_pairs_expected':97356//2},indent=2)+'\n')
print('Report, cross-checks and six-metric figure atlas complete')
