import os
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,csv,math,hashlib
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent
S=json.loads((P/'summary.json').read_text());I=json.loads((P/'integrity.json').read_text())
def read(n):return list(csv.DictReader((P/n).open()))
r=read('per_sample_improved.csv');a=read('modal_alignment.csv');l=read('pair_landscapes.csv');search=read('search_log_coverage.csv');corr=read('gain_complexity_correlations.csv')
def f(x,k):return float(x[k])
def yes(x,k):return x[k]=='True'
def mean(v):return float(np.mean(v))
def ci(v):
 x=np.array(v);rg=np.random.default_rng(20260923);return np.quantile(x[rg.integers(len(x),size=(4000,len(x)))].mean(1),[.025,.975]).tolist()
def fisher(a,b,c,d):
 n=a+b+c+d;m=a+c;k=a+b;den=math.comb(n,k)
 def p(x):return math.comb(m,x)*math.comb(n-m,k-x)/den
 obs=p(a);return sum(p(x) for x in range(max(0,k-(n-m)),min(k,m)+1) if p(x)<=obs+1e-15)
extra={};bounds=[]
for q in l:
 scores=json.loads(q['score_by_tau_null_means_unobserved']);T=int(q['T']);best=max(scores);taus=[i+1 for i,v in enumerate(scores) if abs(v-best)<1e-12];ss=sorted(scores,reverse=True);mt=json.loads(q['matching_taus']);span=max(mt)-min(mt) if mt else None
 bounds.append({'sample_id':q['sample_id'],'task':q['task'],'A':q['A'],'B':q['B'],'best_tau_set':json.dumps(taus),'best_tau_mean_norm':mean(taus)/T,'best_second_tau_margin':ss[0]-ss[1],'first_match_norm':min(mt)/T if mt else None,'last_match_norm':max(mt)/T if mt else None,'match_span_norm':span/T if mt else None,'contiguous_width_norm':max(0,int(q['observed_contiguous_max_n'])-1)/T,'match_n':len(mt),'full_score_tau_n':sum(abs(v-1)<1e-12 for v in scores)})
with (P/'boundary_details.csv').open('w',newline='') as out:
 w=csv.DictWriter(out,fieldnames=list(bounds[0]));w.writeheader();w.writerows(bounds)
for name,rs in [('all',r),('multi_matched',[q for q in r if f(q,'switch_count')>1 and yes(q,'r2_match')])]:
 ids={q['sample_id'] for q in rs};aa=[q for q in a if q['sample_id'] in ids];ls=[q for q in l if q['sample_id'] in ids]
 vals=[f(q,'best_reverse_delta_if_observed') for q in rs]
 extra[name]={'n':len(rs),'selected_best_reverse_mean_delta':mean(vals),'selected_best_reverse_positive_n':sum(x>1e-12 for x in vals),'selected_best_reverse_equal_n':sum(abs(x)<1e-12 for x in vals),'modal_overlap_n':sum(yes(q,'modal_sets_overlap') for q in aa),'modal_n':len(aa),'fragmented_match_pair_n':sum(int(q['r2_match_segment_n'])>1 for q in ls),'fragmented_match_sample_n':len({q['sample_id'] for q in ls if int(q['r2_match_segment_n'])>1}),'last_closer_n':sum(f(q,'best_mean_distance_last_norm')<f(q,'best_mean_distance_first_norm')-1e-12 for q in rs),'first_closer_n':sum(f(q,'best_mean_distance_first_norm')<f(q,'best_mean_distance_last_norm')-1e-12 for q in rs),'recovery_sum_gain_ratio':sum(f(q,'r2_score')-f(q,'sample_best') for q in rs)/sum(f(q,'gain') for q in rs)}
extra['revisit_fisher_two_sided']=fisher(14,2,47,15)
extra['search']={}
for key in ['first_improvement_list_index','first_final_score_list_index','first_selected_sequence_list_index']:
 vals=[f(q,key)/f(q,'candidate_n') for q in search];extra['search'][key]={'mean_fraction':mean(vals),'median_fraction':float(np.median(vals)),'by_half_n':sum(v<=.5 for v in vals)}
# Sensitivity to assigning odd middle chunk to early half; all matching trajectories equally weighted per sample.
sensitivity=[]
for q in r:
 sid=q['sample_id'];sq=json.loads(q['trajectory']);T=len(sq)
 def modes(v):
  c=Counter(v);return {x for x,n in c.items() if n==max(c.values())}
 ee=modes(sq[:(T+1)//2]);ll=modes(sq[(T+1)//2:]);pp=[p for p in l if p['sample_id']==sid and int(p['match_n'])>0];n=sum(int(p['match_n']) for p in pp)
 if n:sensitivity.append(sum(int(p['match_n']) for p in pp if p['A'] in ee and p['B'] in ll)/n)
extra['odd_split_joint_alignment']=mean(sensitivity)
(P/'supplementary_summary.json').write_text(json.dumps(extra,indent=2)+'\n')
all_=S['improved170'];multi=S['native_multi_switch'];mm=S['multi_matched'];un=S['multi_unmatched'];co=corr[0]
text=f'''本次审计仅使用 CPU，分析最新 V2 Dynamic 与其配套完整 R2；未加载模型、未调用 CUDA、未补跑任何推理。

数据依据：Dynamic 为 2026-09-21 合并的 dynamic_old1k_v2，R2 为 mechanism_old1k_v2_20260921（2026-09-22 13:44 完成）。独立核验 1,433 份 Dynamic 来源文件 SHA256、531,113 条候选记录、170 份 R2 结果、97,356 个唯一一次切换候选；其中 7,733 条复用记录逐条与当前 V2 Dynamic 分数和预测哈希对齐。全部通过。旧 V1 R2 未作为统计输入。

| 核心指标 | 最新结果 |
|---|---:|
| Dynamic 改善样本 | 170 |
| Native 一次切换 / 多次切换 | 92 / 78 |
| R2 匹配 Native | 154/170（90.59%） |
| 多次切换可被 R2 匹配 | 62/78（79.49%） |
| 完整 R2 仍未匹配 | 16/78 |
| 逐样本 gain recovery 均值 | 90.59%，任务内分层 bootstrap 95% CI [85.88%, 94.71%] |
| 总增益恢复比（先求和再相除） | {extra['all']['recovery_sum_gain_ratio']:.2%} |

恢复率 (R2−SampleBest)/(Dynamic−SampleBest) 不裁剪。本次实测 154 个为 1、16 个为 0；中位数、Q25、Q75 均为 1，≥0.9 和 ≥0.5 均为 154/170。16 个未匹配样本的最优 R2 均等于各自 SampleBest，并非部分恢复。以上结论条件于 Dynamic 有提升的样本，不能外推为未筛选总体成功率。

切换时间与解的多重性：62 个可压缩的多次切换样本中，50 个有多个同分 R2，12 个仅有一个；匹配候选数中位数 4。然而只有 22/62 存在同一动作对的相邻成功切换点，其余 40/62 的成功点在各动作对内均孤立。存在多个解，不等于切换时间宽容。{extra['multi_matched']['fragmented_match_sample_n']} 个样本含不连续匹配集合；表格分别记录最早/最晚点、跨度和最长连续段，绝不把跨度当连续平台。当前证据更支持“低切换次数，但常有时间敏感性”，不能据此断言 1K→2K 性能变化的原因。

动作对齐：对每个样本的全部同分 R2 等权，再对样本等权；early 用前 floor(T/2) 个 chunk，late 用剩余 chunk，众数并列取集合。62 个匹配的多次切换样本，early 对齐 {mm['alignment']['early_alignment']:.2%}、late {mm['alignment']['late_alignment']:.2%}、同时对齐 {mm['alignment']['joint_alignment']:.2%}；只有 2/62 至少存在一个同时对齐的候选。全体匹配样本中 {extra['all']['modal_overlap_n']}/154 的 early/late 众数集合有交集，长末段经常同时支配两个半段。奇数中间 chunk 改分给 early 后，全体联合对齐为 {extra['odd_split_joint_alignment']:.2%}。因此这个半段众数指标不能支撑“保留早期状态形成动作→晚期利用动作”的叙事；低对齐也不构成该内部机制的反证。

边界位置：对全部最优 tau 平均距离，匹配的多次切换样本距离首次切换为 {mm['metrics']['best_mean_distance_first_norm']['mean']:.3f}T，距离末次切换为 {mm['metrics']['best_mean_distance_last_norm']['mean']:.3f}T；{extra['multi_matched']['last_closer_n']}/62 更接近末次，{extra['multi_matched']['first_closer_n']}/62 更接近首次。末次切换边界与最后稳定 run 起点是同一位置的两种索引：若边界为 tau，稳定 run 从第 tau+1 个 chunk 开始，不应当作两个独立发现。最优 tau 的确定性代表采用最早 tau，存在向早期偏置，因此表格同时保留最优集合。

方向性：逐一配对 A^tau B^(T−tau) 与 B^(T−tau) A^tau，保留相同动作次数和切换数。完整枚举有 48,678 个无序配对。全体样本先各自平均再平均，{all_['available_reverse_mean_unequal_fraction']:.2%} 的配对分数不同，绝对分差均值 {100*all_['available_reverse_mean_abs_delta']:.2f} 个百分点。62 个匹配的多次切换样本，确定性最优代表优于反转 {extra['multi_matched']['selected_best_reverse_positive_n']}/62，平均差 {100*extra['multi_matched']['selected_best_reverse_mean_delta']:.2f} 个百分点。这说明已观测轨迹有顺序敏感性，但“选最优再比较反向”的优势有选择偏差；不能把该均值当无偏因果效应或宣称某个固定层级方向普遍更好。全体配对同时含两种方向时，有符号均值可因对称性抵消，故无选择描述使用绝对差。

16 个失败样本：平均切换数 4.44，匹配组为 3.42；平均 return count 2.13 对 1.31。出现 revisitation 的比例为 14/16 对 47/62，Fisher 双侧 p={extra['revisit_fisher_two_sided']:.3f}。多数成功组也会返回旧动作，证据不足以宣称“R2 失败由 action revisitation 导致”。这里的失败仅指固定动作集合、chunk 网格与既定评分下完整一次切换集合未达到 Native，未确定最少需要两次还是更多次切换。

收益与复杂度：170 个改善样本的 gain 与 switch count 的 Spearman rho={float(co['rho']):.3f}，任务内分层 bootstrap 95% CI [{float(co['ci_low']):.3f}, {float(co['ci_high']):.3f}]。没有观察到明确单调关系；这不是证明独立性。完整指标和任务内相关见 gain_complexity_correlations.csv，gain 分箱见 gain_by_switch_count.csv。归一化首次切换位置容易受任务、T 和搜索位置顺序混杂，不能将其较高相关直接解释为机制。

预测一致性：154 个分数匹配样本中，152 个至少有一个 R2 与 Native 预测哈希一致；确定性最优代表为 144/154。多次切换子集分别为 60/62 和 59/62。单次输出哈希相同只能说明这次记录的预测一致，不能升级为隐藏状态或普遍行为等价。92 个 Native 本身一次切换的样本属于结构上容易匹配的子集，并且 R2 允许复用同一条日志，不能把复用匹配当独立重放验证。

跨任务：9 个有改善的任务都有 R2 匹配，但匹配比例并不一致（例如 multikey3 为 12/16，VT 为 71/77）；另 3 个任务无改善样本。逐任务结构指标见 per_task.csv。transition_enrichment.csv 使用每样本总权重 1 的匹配动作对计数，与 Native 动作占比构造的、条件于 A≠B 的边际独立基线比较。该基线仅为描述性参照，未校正候选筛选或做方向显著性检验，不能据此写跨任务固定方向结论。

搜索过程：新版 beam_search 依次追加评估并按 sequence_hash 保留首次插入顺序，候选列表可用于“首次发现的去重候选序号”，但缓存命中、七个 constant seed 和最终强制 replay 使它不等于原始调用次数或墙钟时间。没有显式 replay_index/parent，不能重建完整父子搜索树。首次提升的归一化序号中位数 {extra['search']['first_improvement_list_index']['median_fraction']:.3f}，首次最终分数中位数 {extra['search']['first_final_score_list_index']['median_fraction']:.3f}，最终选中序列首次出现中位数 {extra['search']['first_selected_sequence_list_index']['median_fraction']:.3f}。这些仍是改善样本上的条件统计，不足以说明随机搜索也容易发现解。

可用于论文的保守表述：在最新 V2 的 78 个多次切换 Dynamic 改善样本中，完整一次切换搜索可匹配 62 个的分数，其中 60 个还存在预测哈希一致的解。这些低切换解通常不唯一，但匹配集合往往缺少相邻成功边界；低切换复杂度与精细时间需求可以共存。数据不支持直接把前后半段众数对齐、动作返回或最优反转差写成内部状态形成/利用机制的因果证明。

复现：使用现有 /home/zonghan/conda_envs/ttt_phase_c_v1/bin/python 依次执行本目录 audit.py 和 report.py。脚本显式屏蔽 GPU，仅使用标准库和 NumPy，计算线程数为 1。统计容差 1e-12，随机种子 20260923，均值 bootstrap 4,000 次、相关 bootstrap 2,000 次；相关探索未进行多重比较校正。原始文件保持只读。input_provenance.json 保存输入哈希；integrity.json 保存覆盖与来源检查。
'''
(P/'CPU_AUDIT_REPORT.md').write_text(text)
# Detect source edits during this audit without rerunning scientific calculations.
prov=json.loads((P/'input_provenance.json').read_text());bad=[]
for path,v in prov.items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=v['sha256']:bad.append(path)
assert not bad,bad
(P/'source_preservation.json').write_text(json.dumps({'status':'PASS','inputs_rehashed':len(prov),'changed':bad},indent=2)+'\n')
# Add the read-only search implementation to provenance of search interpretation.
f=Path('/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918/runtime/executor.py')
(P/'search_order_provenance.json').write_text(json.dumps({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'inspected_function':'beam_search','lines':'202-216','executed':False},indent=2)+'\n')
print(json.dumps(extra,indent=2))
