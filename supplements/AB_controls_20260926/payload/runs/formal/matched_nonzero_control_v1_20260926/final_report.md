MAGNITUDE_MATCHED_CONTROL_STATUS=PASS
COHORT_N=157
COHORT_EXACT=true
N_SEEDS=5
NATIVE_REPLAY_PASS=true
MAGNITUDE_MATCH_MEDIAN_REL_ERROR=0.0007475710967875704
MAGNITUDE_MATCH_MAX_REL_ERROR=0.01173097268340272
MATCH_WITHIN_1PCT_RATE=0.9994269340974212
MEAN_NATIVE=93.24840764331209
MEAN_DELETION=54.30785562632695
MEAN_MATCHED_RANDOM=58.84968152866242
NATIVE_MINUS_RANDOM_PP=34.39872611464968
NATIVE_MINUS_RANDOM_CI95=[31.500636942675158, 37.24506369426751]
RANDOM_MINUS_DELETION_PP=4.541825902335457
RANDOM_MINUS_DELETION_CI95=[1.6708704883227177, 7.347388535031845]
FINAL_INTEGRITY=PASS
FORMAL_COMPLETE=true
PROTOCOL_CLOSEST_FALLBACK_VALIDATED=true
STRICT_WITHIN_1PCT=false

# Magnitude-matched nonzero control

Completed 157/157 examples. Statistical unit is example; average five seeds within branch, then branches within example. Task tables are descriptive only; score columns in task table are percentages, example table scores are fractions. Equality tolerance is 1e-12 score units.

A1 is copied verbatim from the historical SUMMARY.json; historical artifacts are read-only. A2/A3 use paired task-stratified 20,000-draw percentile bootstrap with frozen seed and cohort order.

Magnitude policy: the user permits the closest evaluated deterministic scalar when BF16 quantization misses 1%. CHECK_A9 validates that policy; STRICT_WITHIN_1PCT separately reports the numerical target. Exceptions: 1; maximum relative error 0.01173097268340272; within-1% fraction 0.9994269340974212. No seeds, directions, cohort or calibration algorithm changed. Closest means best among evaluated scalar candidates, not proven global scalar optimum.

Interpretation is conditional on the permitted closest-match BF16 control and its reported residual norm errors; this is numerically limited evidence, not an execution failure or a claim that every norm is within 1%.

CASE 1 under the stated closest-match policy: Matched random exceeds deletion, supporting greater disruption from deleting the trajectory-formed offset than from these approximately matched nonzero perturbations. Residual norm mismatch must accompany this conclusion; it does not establish a unique causal mechanism.

Audit checks:

- CHECK_A1: PASS
- CHECK_A2: PASS
- CHECK_A3: PASS
- CHECK_A4: PASS
- CHECK_A5: PASS
- CHECK_A6: PASS
- CHECK_A7: PASS
- CHECK_A8: PASS
- CHECK_A9: PASS
- CHECK_A10: PASS
- CHECK_A11: PASS
- CHECK_A12: PASS

## 中文结论与数值解释

完整执行157个样本、349个target-layer分支、5个冻结seed，共1745次正式随机扰动评估。Smoke为5个样本、65次扰动、13次同seed重复、13次Deletion重放。

| 条件/差值 | 分数或pp | 95% CI (pp) |
|---|---:|---|
| Native | 93.248408 | — |
| Selective Deletion | 54.307856 | — |
| Matched Random | 58.849682 | — |
| Native − Deletion（直接引用正式A1） | 38.940552 | [35.167728237791934, 42.579883227176204] |
| Native − Random（A2） | 34.398726 | [31.500636942675158, 37.24506369426751] |
| Random − Deletion（A3） | 4.541826 | [1.6708704883227177, 7.347388535031845] |

Random相对Native：149个更低、8个相等、0个更高。Random相对Deletion：74个更高、51个相等、32个更低。相等阈值为1e-12 score units。

A3置信区间为正，支持在此冻结改善cohort与允许的数值匹配协议下，删除trajectory-formed offset比同幅度随机非零扰动平均更具破坏性。但同幅度随机扰动仍带来34.40pp的显著下降；本结果不能被写成“损害主要由特定方向导致”，也不证明该方向是唯一causal mechanism。受改善样本筛选条件、固定五个随机种子及BF16量化影响，不能推广为全任务无条件因果效应。

幅度相对误差（以下均为百分数）：median=0.074757%，mean=0.110044%，max=1.173097%。严格<=1%为1744/1745（99.942693%）。唯一例外为ruler_qa_squad_16k:0161、L18、seed20260930，目标1.317986857429787e-7，实际1.3025255896081944e-7，误差1.173097%。固定方向搜索达到32次binary上限后保留已评估的最接近非零解；这不是全局不可达的数学证明。不得称全部扰动满足1%。

## 执行修订与完整性

最初runner把1%设为硬停止条件，严于用户允许的closest fallback，完成72个样本后停止。PROTOCOL_AMENDMENT_01.json记录了修正，仅改变是否继续执行，不改变Gaussian方向、seed、scalar calibration算法/迭代上限、cohort、轨迹、边界、prefix KV、generation、scoring或统计协议。旧文件和停止证据保留，72份已完成结果原样复用。原历史正式实验文件没有修改。最终PASS指允许最接近解协议通过；STRICT_WITHIN_1PCT=false明确保留。

当前执行入口为run_control_closest.py、pipeline_closest.py，汇总入口为summarize_closest.py；原始版本仍在目录中供审计。查看pipeline_closest_status.json确认COMPLETE。aggregation_smoke_check/与aggregation_closest_check/是部分数据汇总测试，不是正式结果。
