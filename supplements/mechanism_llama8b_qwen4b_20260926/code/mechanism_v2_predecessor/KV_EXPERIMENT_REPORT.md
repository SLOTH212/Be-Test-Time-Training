# V2 KV 干预实验报告与跨模型复现说明

日期：2026-09-26。对象：最终论文 Qwen3-1.7B、旧训练权重、1K chunk、Reference Executor V2。此次只读取既有 GPU 实验记录并独立 CPU 重算，没有重新执行模型。本文中的 K 指整组 attention Key/Value cache，不是只干预 Key。P 轴不包含在本实验中。

## 1. 问题与主结论

研究问题：在 Native Dynamic 轨迹的指定干预边界，将候选层快权重恢复成 Native 状态后，前缀 KV 来源是否仍影响后续表现？

在预先按轨迹结构筛出的 165 个改善样本上，K0 使用 OFF-prefix 产生的前缀 KV，K1 使用 Native 前缀 KV；二者在各候选层边界注入相同的 Native 快权重，并使用相同的后缀动作。K0 平均分 78.7778，K1 为 93.3333，配对差值为 **+14.5556 个百分点，95% CI [10.5354, 18.7677]**。47 个样本提高，118 个相等，0 个降低。K1 与 Native 在已记录 trace、预测 hash、得分上均 165/165 闭合。

支持的解释：仅恢复边界快权重不足以在所有样本上重建 Native 后续行为；Native 前缀 KV 携带了对重建行为有用的历史信息。该差值是前缀缓存替换的**总下游效应**，包含后续激活与快权重更新重新计算的影响，不能称为独立、直接的 KV 效应，也不能推论所有 Dynamic 收益均来自 KV。

## 2. 冻结模型与执行条件

- 权重：`/home/USER/ttt/models/qwen3_1p7b_stage2`，已训练 Stage-2 的旧 1K 模型，不是未经训练的 Qwen3 Base，也不是后来训练的 2K 模型。
- 权重 SHA256：`ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f`。
- 16K RULER 输入，实际 token 数按冻结 tokenizer 和样本计数；`add_special_tokens=True, truncation=False`。完整 chunk 数 `T = token_count // 1024`，不是统一假定为 16。
- chunk=1024；候选写入层 `[0,6,12,18,24]`，动作 `OFF,L0,L6,L12,L18,L24,ALL`。ALL 仅指这五个候选层。
- BF16、SDPA、greedy：`do_sample=False, num_beams=1, use_cache=True`；生成上限使用每个样本冻结的 `max_new_tokens`。
- 全部动作都保持五个候选层的功能式路径；动作只控制写入。完整 chunk 先读取当前快权重得到输出，再提出并应用更新；尾部不足一个 chunk 的 token 和生成阶段不写入。
- TTT 更新实现、学习率/裁剪、tokenizer、score 函数必须使用冻结 runtime，不能仅靠上述参数自行换成另一套更新公式。
- 分支从干净状态完整重放，不拼接 Native 已存储的后续更新量。随机种子 20260921。

## 3. 样本选择与边界：170 → 165 → 157

先冻结当前 V2 Dynamic 严格超过该样本自身七个 V2 Fixed 分数最大值的 170 个改善样本。此集合有参考答案评分参与选择，属于条件性机制诊断，不能作为无监督部署收益估计。当前非 CWE 12-task 主实验共有 6000 个样本，本报告的 165 个样本不代表所有 6000 个样本。

对动作数组 `a[0:T]` 定义 `switches = [i for i in range(1,T) if a[i] != a[i-1]]`。选择规则：

```python
first_switched_chunk = switches[0] + 1  # 一基 chunk 编号
late = a[T // 2:]                       # 奇数 T 时该半段较长
eligible = first_switched_chunk / T <= 0.25 and modal_count(late) / len(late) >= 0.8
tau = switches[-1]                      # 最后一次切换；零基 suffix 起点
boundary_chunk = tau + 1
prefix_tokens = tau * 1024
control_sequence = ['OFF'] * tau + a[tau:]
```

满足结构条件的 165 个样本是 K 主 cohort。边界取**最后一次切换**，不是第一次切换，也不是固定输入中点。157 个 nontrivial 样本再要求 `a[:tau]` 至少有一个非 OFF 动作；另外 8 个是全 OFF 前缀。不能按 K0 是否掉分重新筛选 cohort。本次 CPU 审计逐样本重新计算这些规则，与冻结 COHORT 完全一致。

## 4. 四个分支的定义

| 分支 | 前缀动作 | 候选层边界快权重 | 前缀 KV | 后缀动作 |
|---|---|---|---|---|
| Native | 原 Dynamic | 自然形成的 Native | Native | 原 Dynamic |
| Control | 全 OFF | OFF-prefix 自然形成 | Control | 原 Dynamic |
| K0 | 全 OFF | 五层注入 Native 边界权重 | Control | 原 Dynamic |
| K1 | 全 OFF | 五层注入 Native 边界权重 | 全部 28 个 attention 层换为 Native | 原 Dynamic |

K0/K1 只在对应层的干预边界匹配快权重，**后续权重没有锁定为相同**。Control 在当前 V2 core 记录中确实存在；不要沿用旧材料中“没有保存 Control”的说法。该设计不是一个完整的 W×KV 2×2 因子实验，未包含“Control 权重 + Native KV”的第四因子组合；Native 是原路径参考。

## 5. 实际代码执行顺序：必须按此理解

当前实现采用完整 prompt prefill，TTT 候选 MLP 内部再按 chunk 顺序计算。它不是在整个模型上按 chunk 推进到某个全局时间点后暂停。

1. 干净重放 Native，核对得分及预测 hash 与冻结 Dynamic winner 一致。
2. 带 trace 和自定义 cache 再运行 Native。每个候选 MLP 到达 `j == tau` 时，在该层 suffix chunk 读取/写入之前，保存当前实际快权重 `_boundary_before`，不是保存待应用 delta。
3. Native 完整 generate 结束后，从全部 28 层 cache 取 `keys[..., :tau*1024, :]` 和对应 values，detach 后复制到 CPU 作为 donor。只取前缀，不包含 suffix 或生成 token。
4. 干净重放 Control。
5. 干净重放 K0：使用 Control 动作序列，每个候选 MLP 到 `j == tau` 时，注入该层 Native 边界快权重，然后照常重新计算后缀。
6. 干净重放 K1：同 K0，但每个 attention 层第一次多 token `cache.update` 时，在父类更新缓存后、attention 消费返回 K/V 前，原位替换前 `tau*1024` 个位置为 Native donor。每层仅替换一次，suffix KV 保留该分支自己的计算。
7. 检查每层替换 slice 的 hash 与 donor 完全相等，且全 28 层都有替换记录；检查 K1/Native trace 和预测闭合。最后再次干净重放 Native，检查重复性。

Qwen3 中 cache Key 已经过 k_norm 和 RoPE；Value 使用 cache 原生布局。q/k/v projection trace hook 记录的是投影输出，**不是**归一化和 RoPE 后的 cache Key。跨模型时必须适配真实缓存布局、KV head 数、position/RoPE 和 cache API，不能把 projection hook 输出直接当 donor。

替换发生在完整 prefill 中，因此可能改变 K1 的前缀区域内部计算；本实验要求的是所记录 suffix 闭合，不要求 K1 的全部前缀激活与 Control 或 Native 相等。若改成“加载全模型边界快照，只重放 suffix”，这是待验证的另一种执行方法，不能声称它已被本实验验证等价。

## 6. 正式结果与补充核算

得分为逐样本 scorer 值的均值×100；不是二元答对率。pp 指百分点，不是相对百分比。

| cohort | N | Sample Best | Native | Control | K0 | K1 | K1−K0 (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 主 K cohort | 165 | 41.1515 | 93.3333 | 46.7172 | 78.7778 | 93.3333 | +14.5556 | [10.5354,18.7677] |
| nontrivial prefix | 157 | 41.1677 | 93.2484 | 44.2569 | 77.9512 | 93.2484 | +15.2972 | [11.2312,19.7452] |
| 全 OFF prefix | 8 | 40.8333 | 95.0000 | 95.0000 | 95.0000 | 95.0000 | 0.0000 | [0,0] |

Control 列和全 OFF 子组是在本次报告中从已保存 core 独立补算，没有新增模型执行。

| 主 cohort 的闭合检查 | 通过数 |
|---|---:|
| K0 与 Native 已记录 trace 完全相同 | 110/165 |
| K0 与 Native decoded prediction hash 相同 | 118/165 |
| K1 与 Native 已记录 trace 完全相同 | 165/165 |
| K1 与 Native prediction hash、score 相同 | 165/165 |
| K1 28 层 prefix K/V 替换记录与 donor hash 一致 | 165/165 |
| K0、K1 五层注入 hash 与 Native 对应边界权重一致 | 165/165 |

K0 有 8 个样本预测相同而已记录 trace 不同，说明“答案相同”不能代替内部状态闭合检查。

置信区间：20,000 次配对、按 task 分层的 bootstrap，seed=20260921，NumPy default_rng，保持每类样本数不变；每次对样本差值重采样后按全样本数取均值。任务按字典序，任务内保持 COHORT 原顺序，每批 1000，线性 2.5%/97.5% quantile。随附 CPU 脚本复现该顺序，不直接运行原 summarize.py，避免覆盖历史输出。

## 7. 已验证范围与不可扩大主张

trace 记录覆盖：全部 28 层的首个 suffix 完整 chunk 的 attention input、q/k/v projection、attention output、MLP input/output、residual；存在 remainder 时另记尾部；final norm 的整个 suffix；第一次 forward 返回的 logits。生成阶段单 token 的这些层级 hook 被跳过，另用最终 decoded prediction hash 检查输出。因此不得写成“所有生成步、全部 KV、所有内部状态逐元素均已验证相同”。

分支 `base_unchanged` 对应的运行期 base_hash 检查仅覆盖五个候选 MLP 的 down_proj 权重，不等于每个分支对完整 checkpoint 所有 tensor 都重做 hash。

K1 精确闭合不证明一般性的 full replay 与 arbitrary prefix-state reuse 等价；也不证明标准路径与功能式路径在全部情形下数值等价。V2 将动作统一到同一功能式路径以避免该路径选择混淆。当前 adapter 的这些结果不能替代对新模型、新精度、新 attention backend 的执行有效性检查。

## 8. 迁移到其他模型的实施与验收要求

1. 冻结 checkpoint/tokenizer/scorer/数据、chunk、dtype、backend、动作集合、层映射和生成设置；记录代码与数据 hash。层数不同，须明确新的候选层，不能机械使用超出范围的原层编号。
2. 先验证先读后写、未来动作不改变共同前缀、强制零更新时动作路径一致、重复执行一致、样本间状态清空、tail/generation 不写入。需保存实际运行报告，不能只引用 1.7B 已有报告。
3. 运行本模型自己的 Fixed 与 Dynamic，冻结获胜轨迹和改善 cohort，再按同一结构规则筛选 K cohort。若直接复用 1.7B 的 165 个 sample ID，必须标成“固定样本跨模型对照”，不能称本模型自身改善 cohort。
4. 适配真实 cache API；在 attention 消费之前替换 prefix，按本模型所有 attention 层核验层号、长度、位置、shape、dtype 和 hash。GQA/MQA 模型以原生 KV head 布局为准。
5. 运行 Native/Control/K0/K1 和 Native repeat，记录每个样本 score、预测 hash、边界、动作、注入前后权重 hash、KV donor/替换 hash、trace、环境和源文件 hash。保留失败样本及失败原因，不能删除不闭合样本后只报成功率。
6. 先在固定环境争取精确闭合；若更换硬件/backend 无法 bitwise 闭合，另行预注册数值误差指标与阈值，同时报告预测/得分一致性，不能事后放宽门槛沿用“exact”。
7. 以本模型实际结果报告 N、均值、配对差、分层 CI、正/平/负样本数及闭合率。1.7B 的 +14.56 pp 是参考结果，不是新模型必须达到的验收目标。

## 9. 原始来源与交付文件

本机 authority：`/home/USER/ttt/runs/formal/mechanism_old1k_v2_20260921/`。

- `run.py:22–94`：trace、cache 替换、Native/Control/K0/K1 分支。
- `runtime/executor.py`：统一写入路径、边界快权重注入、clean replay、计分。
- `runtime/code/inference_model/hf_qwen3/modeling_qwen3.py:280–301`：RoPE、cache.update 与 attention 顺序。
- `COHORT.json`：170 个改善样本，含 eligibility、边界、轨迹和 Dynamic 来源 hash。
- `PROTOCOL.json`、`SOURCE_HASHES.json`、`samples.jsonl`、`core/*.json`：协议、源码、输入及逐样本原始记录。

已核对 GitHub 版本的固定 commit：`217501bca838542083851f0e50c0c8c2995f2a4b`。
[GitHub V2 发布目录](https://github.com/SLOTH212/Be-Test-Time-Training/tree/217501bca838542083851f0e50c0c8c2995f2a4b/dynamic-ttt-v2-paper-20260924)

仓库子目录 `dynamic-ttt-v2-paper-20260924/` 内：`experiments/mechanism/` 对应机制源码与协议；`data/mechanism/core.jsonl.gz` 为打包的 core 原始记录；`data/derived/MECHANISM_SAMPLE_PAIRS.tsv` 与 `results/writer/TABLE_MECHANISM_V2.tsv` 为导出表。GPU 原脚本含原机器路径，移植需先重定向到独立目录并调整数据/模型路径，不能直接在历史目录运行覆盖记录。模型权重与完整运行环境需另行准备。

本次交付目录内：

- `KV_EXPERIMENT_REPORT.md`：本报告。
- `KV_PER_SAMPLE.tsv`：165 行逐样本分数、边界、动作和闭合指标。
- `KV_SUMMARY.json`：总体、157/8 子组、按任务统计。
- `recompute_kv.py`：只读历史数据、写入本目录的 CPU 核算及断言脚本；原始数据根目录在脚本变量 R，迁移需调整。
- `SOURCE_SHA256.json`：本次读取的 core 和关键源码/协议文件 hash。

复算命令（本机，不加载模型）：

```bash
taskset -c 8-23 /home/USER/conda_envs/ttt_phase_c_v1/bin/python /home/USER/ttt/audits/kv_v2_replication_handoff_20260926/recompute_kv.py
```
