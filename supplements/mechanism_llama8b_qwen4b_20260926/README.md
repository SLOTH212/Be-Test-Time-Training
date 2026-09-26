# 机制实验补充：Llama-3.1-8B 与 Qwen3-4B（RULER 32K / 4K chunk）

本目录归档 Llama-3.1-8B 与 Qwen3-4B 的机制实验（V4，GPU0–7 八卡正式运行）。任务于 2026-09-26 15:51:57（UTC+8）完成，15:53:59 独立核验 **PASS**，同日导出。它是 `supplements/` 下的独立补充目录，与 `llama_github_20260925/`、`qwen3_4b/`、`supplements/AB_controls_20260926/` 并存，不替换其中任何内容。

## 对象与设置

| | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| 模型 | Stage2 FINAL `llama31_8b_stage2_15m_32k_gpu01234_final_v3` | Stage2 FINAL `qwen3_4b_stage2_30m_32k_final_v1` |
| 父 Dynamic 结果 | `llama_github_20260925/results/dynamic_v2_no_cwe/` | `qwen3_4b/results/dynamic_v2/` |
| Cohort：Dynamic 严格优于 Sample Best 的全部样本 | 39 | 85 |
| Reset / KV 合格样本 | 30 | 69 |
| Selective 合格样本（目标分支 / SHAM） | 29（64 / 22） | 62（117 / 51） |
| R2 候选数 | 12,992 | 28,560 |

两模型共同设置：上下文 32,768，chunk 4,096；TTT 层 0、6、12、18、24、30；动作 OFF、L0、L6、L12、L18、L24、L30、ALL；12 个任务，不含 CWE。任务名中的 `_16k` 是历史名称，实际为 32K 配置。

## 机制定义

完整协议见 `code/mechanism_v4/PROTOCOL.md` 与 `code/mechanism_v2_predecessor/PROTOCOL.md`。

- **Native**：重放父 Dynamic 的最优动作序列，必须与已提交的 Dynamic 分数和解码预测 hash 完全一致。
- **Reverse**：把完整 chunk 的动作序列整体反转。
- **Reset**：在最后一段恒定动作的起点（最后一次切换边界）之前，把六层有效快权重全部恢复为 checkpoint 权重，保留前缀上下文。
- **KV**：在同一边界，K0 和 K1 都向六层注入相同的 Native 边界快权重，后缀动作也相同。K0 使用 OFF 前缀 Control 重算出的前缀 KV；K1 把所有 attention 层的前缀 KV 换成 Native 的。K1−K0 是换入 Native 前缀 KV 的**总下游效应**。要求 K1 的预测、最终状态 hash 和已记录的 suffix trace 与 Native 完全一致。仅在 Reset 合格样本上评估，不含 P 因子。
- **Selective**：迁移自 1.7B 的历史 deletion_plan，适配为六个候选层和 4K chunk。在最后一次切换边界，把选定层的有效 down-projection 快权重恢复为 checkpoint，即删除该层已累积的参数状态；保留 Native 前缀 KV 和其他层状态，后续动作照常执行（状态可以重新累积）。
  - 前缀含 ALL 时，六层逐一删除（leave-one-layer-out）并在样本内取平均；
  - 否则删除前缀中第一个非 OFF 动作对应的层；
  - 前缀没有任何写入的样本不合格。
- **SHAM**：删除前缀中从未写入的最低层。该层状态本来为零，删除应当没有任何影响。
- **R2**：只允许切换一次。对所有有序且不同的动作对（8×7=56）和所有内部边界逐样本穷举，不提前停止。平局依次按最小边界、A 的规范动作顺序、B 的规范动作顺序、序列 hash 选取。

## 结果

分数为逐样本 scorer 值的均值 ×100；差值单位是百分点。

| 机制 | Llama-3.1-8B | Qwen3-4B |
|---|---|---|
| Dynamic（Native）/ Sample Best | 100.00 / 40.77 | 97.02 / 37.14 |
| **R2 最优** | **96.58**；37/39 追平 Dynamic，2 个低于 | **93.51**；79/85 追平 Dynamic，6 个低于 |
| R2 保留的 Dynamic 相对 Sample Best 增益 | 94.2% | 94.1% |
| Reverse | 39 个样本：100.00 → 44.87，平均降 **55.13**；降/平/升 36/3/0 | 85 个样本：97.02 → 43.02，平均降 **54.00**；76/9/0 |
| Reset | 30 个样本：100.00 → 47.39，平均降 **52.61**；26/4/0 | 69 个样本：96.33 → 47.25，平均降 **49.08**；58/11/0 |
| Selective | 29 个样本：100.00 → 49.26，平均降 **50.74**；25/4/0 | 62 个样本：96.45 → 43.91，平均降 **52.54**；59/3/0 |
| SHAM | 22/22 无变化 | 51/51 无变化 |
| **KV（K1−K0）** | 30 个样本：99.33 → 100.00，**+0.67**；升/平/降 1/29/0 | 69 个样本：88.55 → 96.33，**+7.78**；10/59/0 |

- R2 未追平 Dynamic 的样本：
  - Llama：`ruler_fwe_16k:0476`、`ruler_qa_squad_16k:0461`；
  - Qwen：`ruler_niah_multivalue_16k:0056`、`ruler_qa_squad_16k:0044`、`ruler_qa_squad_16k:0089`、`ruler_vt_16k:0064`、`ruler_vt_16k:0235`、`ruler_fwe_16k:0432`。
- KV 差值非零的样本：
  - Llama 只有 `ruler_vt_16k:0211`（+20）；
  - Qwen 有 10 个：`ruler_qa_squad_16k:0370`、`ruler_qa_squad_16k:0110`、`ruler_niah_multikey_3_16k:0217` 为 +100，`ruler_fwe_16k:0432` 为 +66.67，`ruler_niah_multivalue_16k:0056` 为 +50，其余在 +20 到 +25 之间。
- 样本来源：Llama 的 39 个样本来自 7 个任务（VT 12、SQuAD 9、HotpotQA 7、FWE 6、multivalue 3、multikey_1 1、multiquery 1）；Qwen 的 85 个样本来自 10 个任务（multivalue 16、SQuAD 14、VT 12、multiquery 11、multikey_3 10、FWE 8、HotpotQA 8、multikey_1 3、multikey_2 2、single_2 1）。

逐样本数据在 `results/<model>/per_sample.json`，全部 R2 候选在 `results/<model>/r2_candidates.csv`。以上数字直接统计自服务器上的最终 `AGGREGATE.json`；其中 R2、KV、Selective 的汇总与服务器端最终审计 `run_records/v4/FINAL_INDEPENDENT_AUDIT.json` 一致。

## 解释与限制

1. **条件性诊断，不是全 benchmark 结果。** Cohort 以"Dynamic 严格优于 Sample Best"为条件选出。Sample Best、Dynamic 和 R2 都用参考答案评分做选择，属于 oracle 搜索。以上数值只适用于各自的筛选样本，不能写成整个 RULER 上的平均分或因果效应。
2. **方向一致。** Reverse、Reset、Selective 在两个模型上都没有任何样本升分；SHAM 全部零变化；KV 没有任何样本降分。
3. **KV 效应随模型不同。** 4B 为 +7.78，8B 只有 +0.67（仅一个样本）。作为参照，1.7B（1K chunk，165 个样本）为 +14.56（见 `code/mechanism_v2_predecessor/KV_EXPERIMENT_REPORT.md`）。KV 边界固定为最后一次切换；8B 的小效应可能与边界位置有关，但本次没有检验这一点。曾提出过四边界扫描，在正式启动前撤回，本目录没有任何四边界结果。
4. **4K 适配的资格规则。** 严格的 1.7B 规则（首次切换位置 / T ≤ 0.25）在这些 6–7 个 chunk 的 prompt 上选出 0 个样本，因此使用 resolution-aware 规则：首次切换位置 ≤ max(2, ⌊N/4⌋)，后半段 modal 占比 ≥ 80%（`code/dependencies/src/reset_resolution_v2/candidate.py`）。得到的合格数为 30 / 69。
5. **一次切换不能覆盖全部收益。** 未追平的样本在原 Dynamic 中使用了多次切换。
6. **Trace 覆盖范围。** 已记录的是 prefill 阶段首个 suffix chunk 与 remainder 的各层 attention / MLP 输入输出、q/k/v 投影和残差，以及 final norm 和返回的 prefill logits。生成阶段的逐 token 激活没有记录，生成输出用解码预测 hash 核对。
7. **部分 8B 结果来自前任任务。** V2 七卡任务先完成了 24 个样本的 Native/Reverse/Reset/KV 和 7,048 条 R2 记录，其中 17 个样本的 V2 任务（含 R2）已全部完成；这些结果核验后导入 V4（`run_records/v4/IMPORT_MANIFEST.json`；`task_receipts_index.json` 中 `origin` 为 `VERIFIED_PREDECESSOR`）。Qwen 的结果全部在 V4 中计算。V4 保持 V2 的 Engine、模型、数据、评分和动作不变，只新增 Selective。
8. **工程记录。** `run_records/v2_predecessor/ERROR_*.json` 是 V2 早期 smoke 中参数注入的类型错误，修复后才正式运行。V3 是已取消的工程 smoke，未收入本目录。最终状态以 V4 的 `PHASE_STATUS.json`（COMPLETE）和 `FINAL_INDEPENDENT_AUDIT.json`（PASS）为准。

## 目录内容

- `results/<model>/per_sample.json`：逐样本记录，包括父 Dynamic 信息、Native/Reverse/Reset/Control/KV、Selective/SHAM 以及 R2 汇总。
- `results/<model>/r2_candidates.csv`：每个样本的全部单切换候选及其分数、预测 hash、schedule hash 和耗时。
- `results/<model>/task_receipts_index.json`：原始任务收据的 SHA256、类型、来源（新算或导入）和状态。
- `code/mechanism_v4/`：V4 的实际代码（`engine.py`、`run.py`、`supplement.py` 等）、`JOB.json`（冻结 cohort、协议字段，以及模型、数据和源码的 SHA256 绑定）、`PROTOCOL.md`、Selective 的 CPU 参考一致性记录。
- `code/mechanism_v2_predecessor/`：V2 的代码、`JOB.json`、协议和 1.7B KV 参考报告。
- `code/dependencies/`：`JOB.json` 中哈希绑定的 32 个依赖源码（导出前逐一核对，全部与绑定 hash 一致）。
- `run_records/v4/`、`run_records/v2_predecessor/`：运行状态、启动与 smoke 记录、导入清单、停止/交接收据、worker 日志和最终审计。
- `provenance/ORIGINAL_RUN_FILES.json`：服务器上 V4 与 V2 运行目录中 3,130 个原始文件（不含锁文件）的大小和 SHA256。
- `SOURCE_MANIFEST.json`：每个导出文件的来源及其原始 SHA256。
- `validation/EXPORT_CHECKS.json`：导出时的脱敏、依赖哈希绑定和敏感信息检查记录。
- `SHA256SUMS.txt`：本目录实际文件的校验值。

## 导出处理与复现

- 服务器根路径替换为 `/path/to/ttt`，账户路径替换为 `/home/USER`。
- `suffix_trace`、`final_hashes`、`boundary_before_hashes`、`boundary_after_hashes`、`donor`、`checkpoint_hashes` 这六类逐层 hash 字典，替换为原值的 canonical SHA256（`{"omitted": true, "entries": n, "canonical_sha256": ...}`）。摘要相同即原字典完全相同，因此各项闭合检查仍可核对。分数、动作序列、预测 hash 和闭合状态均未改动。
- 原始的任务收据、逐样本 `FINAL.json`、`AGGREGATE.json`、导入文件和中间进度没有收入（4B 的 `AGGREGATE.json` 有 183 MB），只在清单中保留 SHA256。
- 不包含模型权重、checkpoint、tokenizer、benchmark 提示和答案、生成文本、运行环境或账户凭据。
- 历史 authority 和 `JOB.json` 中的 hash 指向服务器上的原始文件。脱敏和裁剪会改变文件字节，这是预期行为；校验本目录请使用 `SHA256SUMS.txt`。
- 这是研究归档，不是解压即可运行的发行版。重跑需要自行准备上面两个 Stage2 FINAL 权重、冻结的 RULER 32K benchmark 和对应环境，并把路径重定向到新的目录。不要直接运行历史 manager/launcher，它们会写入原服务器的路径。

## 许可

见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
