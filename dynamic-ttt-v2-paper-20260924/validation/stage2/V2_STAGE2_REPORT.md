# Reference Executor V2 — Stage-2 historical-winner replay audit

Status: COMPLETE. All score means below are percentages; contrasts and confidence intervals are percentage points.

The complete V2 constant family plus each sample's stored historical winner gives a replay gain of 5.433217189 pp, task-stratified 95% percentile CI [4.498228804, 6.415795587], over all 1,435 historical searched samples. The raw winner-minus-Sample-Best contrast is 5.409988386 pp, CI [4.471544715, 6.392595819]; negative raw values are preserved.

## Provenance and execution

The cohort was reconstructed from historical fixed scores and the preregistered 12-task scope, then checked against the selected Dynamic artifacts. It contains 173 improved and 1,262 tied samples in nine tasks. Winners were recovered from stored selected sequences, without reranking candidates or consulting V2 scores. Every seven-constant family and every winner was evaluated anew in Stage-2. Historical files were only read.

V2 source SHA256: `8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c`.
Stage-2 protocol SHA256: `95fd0aa8100d8c53ed0b6e586719436b1be12b6fbc12f250a82addf290cfb8b9`.
Cohort SHA256: `aea199841234fd71436daa97b47dc9c11c26bdb15c1db5a76ee9cc90d5adb02c`.

The four-case bridge matched exactly. All 173 × 8 Stage-1 cells were rerun and matched exact reward, prediction hash, and trajectory. The historical improved split reproduced 153 GT, 19 EQ, 1 LT. Formal execution completed 11480 trajectories with 0 errors and 0 retries; all passed checkpoint and state-isolation checks.

## Estimators and identity clarification

SB_V2 is the maximum of seven fresh Stage-2 constants. HW_V2 is the exact historical selected sequence evaluated under V2. ReplayLB_V2 is max(SB_V2, HW_V2), and ReplayGain is 100 × (ReplayLB_V2 − SB_V2). Raw gap remains 100 × (HW_V2 − SB_V2).

All 1,262 historical tied winners are constant sequences. All 173 historical improved winners are nonconstant. Accordingly, the all-sample estimator uses the **stored historical winner, which may be constant**; it does not add a nonconstant trajectory to each of the 1,435 samples. Constant winners were still independently executed, and all 1,262 duplicate-schedule checks matched the corresponding new constant exactly.

## Results

| Quantity | Percent / pp |
|---|---:|
| Historical SB mean | 34.932636469 |
| Historical Dynamic mean | 41.295005807 |
| Historical gain (pp) | 6.362369338 |
| V2 SB mean | 35.141695703 |
| V2 HW mean | 40.551684088 |
| V2 ReplayLB mean | 40.574912892 |
| V2 raw HW gap (pp) | 5.409988386 |
| V2 replay gain (pp) | 5.433217189 |

Historical Dynamic is the old reward-guided search output; V2 HW is its fixed winner replay; V2 ReplayLB also includes the freshly evaluated complete constant family. These are distinct estimators.

| Historical status | N | V2 GT | V2 EQ | V2 LT | Raw gap pp | Replay gain pp |
|---|---:|---:|---:|---:|---:|---:|
| Improved | 173 | 153 | 19 | 1 | 44.874759152 | 45.067437380 |
| Tied | 1262 | 0 | 1262 | 0 | 0.000000000 | 0.000000000 |

## Task results

| Task | N | Historical improved/tied | GT/EQ/LT | SB % | HW % | ReplayLB % | Raw gap pp | Replay gain pp | Fixed/HW score-change samples |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| ruler_fwe_16k | 312 | 24/288 | 14/297/1 | 65.491453 | 66.880342 | 66.987179 | 1.388889 | 1.495726 | 7/8 |
| ruler_niah_multikey_1_16k | 6 | 1/5 | 1/5/0 | 0.000000 | 16.666667 | 16.666667 | 16.666667 | 16.666667 | 0/0 |
| ruler_niah_multikey_2_16k | 23 | 4/19 | 4/19/0 | 0.000000 | 17.391304 | 17.391304 | 17.391304 | 17.391304 | 0/0 |
| ruler_niah_multikey_3_16k | 132 | 14/118 | 12/120/0 | 0.000000 | 9.090909 | 9.090909 | 9.090909 | 9.090909 | 0/2 |
| ruler_niah_multiquery_16k | 44 | 12/32 | 12/32/0 | 73.863636 | 81.250000 | 81.250000 | 7.386364 | 7.386364 | 0/0 |
| ruler_niah_multivalue_16k | 21 | 8/13 | 8/13/0 | 72.619048 | 83.333333 | 83.333333 | 10.714286 | 10.714286 | 0/0 |
| ruler_qa_hotpotqa_16k | 280 | 18/262 | 10/270/0 | 0.714286 | 4.285714 | 4.285714 | 3.571429 | 3.571429 | 2/6 |
| ruler_qa_squad_16k | 236 | 15/221 | 15/221/0 | 0.000000 | 6.355932 | 6.355932 | 6.355932 | 6.355932 | 0/0 |
| ruler_vt_16k | 381 | 77/304 | 77/304/0 | 65.669291 | 72.440945 | 72.440945 | 6.771654 | 6.771654 | 0/0 |

Task summaries are descriptive; no significance claim is made for small denominators. The aggregate is sample weighted, not a macro average across tasks.

## Historical shifts

Fixed rewards changed in 9 samples (14 cells); winner rewards changed in 16 samples. Fixed prediction hashes changed in 369 cells and winner hashes in 43 samples. Mean SB shift: 0.209059233 pp; mean HW shift: -0.743321719 pp. Changes use exact score inequality. Fixed counts have seven cells per sample; HW counts have one, so cell counts are not directly comparable as rates.

## Statistical protocol and claim limits

NumPy default_rng (PCG64), seed 20260915, 20,000 paired task-stratified draws. Tasks are lexicographically ordered; samples retain the frozen SHA256(sample_id) order inside each task. Each stratum is resampled at its observed sample count, and sums are divided by 1,435. Primary and secondary contrasts share sampled indices. Percentile endpoints use linear quantiles. Input ordering, input hash, draw hash and NumPy version are recorded in V2_BOOTSTRAP_AUDIT.json.

This estimates a conservative historical-winner replay lower bound conditional on the historical searched cohort. It does not estimate V2 Dynamic search or the full 6,000-sample effect: the 4,565 historical ceiling samples have not had their V2 constant family rebuilt. No new candidate search, R2, reverse, reset, deletion, K/P, CWE, cross-model, or random experiment was performed.

The frozen resource guide returns STRONG_REPLAY_RETENTION. Recommendation to separately authorize full V2 Dynamic re-search: YES. This guide is not a manuscript significance threshold. Stage-3 has not been started.

## Required audit output

```text
REFERENCE_EXECUTOR_V2_STAGE2_1435_COMPLETE

[V2_AUTHORITY]
CHECKPOINT_MATCH=YES
MODEL_HASH_MATCH=YES
SCORER_MATCH=YES
TOKENIZER_MATCH=YES
V2_SOURCE_MATCH=YES
CONTEXT=16384
CHUNK=1024
DTYPE=BF16
GPU=NVIDIA GeForce RTX 5090
STAGE2_PROTOCOL_SHA256=95fd0aa8100d8c53ed0b6e586719436b1be12b6fbc12f250a82addf290cfb8b9

[COHORT]
SEARCHED_N=1435
HISTORICAL_IMPROVED_N=173
HISTORICAL_TIED_N=1262
COHORT_RECONSTRUCTION_STATUS=PASS
COHORT_SHA256=aea199841234fd71436daa97b47dc9c11c26bdb15c1db5a76ee9cc90d5adb02c

[STAGE1_BRIDGE]
STAGE1_173_EXACT_REPLICATION=YES
GT_N=153
EQ_N=19
LT_N=1
STATUS=PASS_EXACT

[FORMAL_RUN]
EXPECTED_TRAJECTORY_N=11480
COMPLETED_N=11480
ERROR_N=0
RETRY_N=0
CHECKPOINT_UNCHANGED_N=11480
STATE_RESET_PASS_N=11480

[V2_FIXED7]
EXPECTED_CELL_N=10045
OBSERVED_CELL_N=10045
MISSING_N=0
DUPLICATE_N=0

[V2_HISTORICAL_WINNER]
GT_N=153
EQ_N=1281
LT_N=1
SB_V2_MEAN=35.141695703
HW_V2_MEAN=40.551684088
RAW_HW_MINUS_SB_PP=5.409988386
RAW_HW_MINUS_SB_CI=[4.471544715, 6.392595819]

[V2_REPLAY_LB]
REPLAY_LB_MEAN=40.574912892
REPLAY_GAIN_PP=5.433217189
REPLAY_GAIN_CI=[4.498228804, 6.415795587]
IMPROVED_N=153
ZERO_GAIN_N=1282

[HISTORICAL_IMPROVED_173]
GT_N=153
EQ_N=19
LT_N=1
MEAN_GAP_PP=44.874759152

[HISTORICAL_TIED_1262]
GT_N=0
EQ_N=1262
LT_N=0
MEAN_RAW_GAP_PP=0.000000000
REPLAY_GAIN_PP=0.000000000

[TASKS]
TASK_N=9
TASKS_WITH_POSITIVE_REPLAY_GAIN=9
TASKS_WITH_ZERO_REPLAY_GAIN=0
TASKS_WITH_NEGATIVE_RAW_HW_GAP=0

[HISTORICAL_SHIFT]
FIXED_SCORE_CHANGE_SAMPLE_N=9
HW_SCORE_CHANGE_SAMPLE_N=16
FIXED_PRED_CHANGE_N=369
HW_PRED_CHANGE_N=43
MEAN_SB_SHIFT_PP=0.209059233
MEAN_HW_SHIFT_PP=-0.743321719

[SCIENTIFIC_INTERPRETATION]
CAN_CLAIM_V2_DYNAMIC_LB=NO
CAN_CLAIM_1435_REPLAY_LOWER_BOUND=YES
CAN_CLAIM_PATH_UNIFIED_SEQUENTIAL_UTILITY=YES
CAN_ESTIMATE_FULL_6000_EFFECT=NO

[NEXT_STAGE]
DECISION_BUCKET=STRONG_REPLAY_RETENTION
RECOMMEND_FULL_V2_DYNAMIC_RESEARCH=YES
RECOMMENDED_NEXT_STEP=Request separate authorization to design Stage-3 full V2 Dynamic re-search
```

## CPU serialization recovery

The first CPU summary attempt completed the numerical calculations but failed while serializing a NumPy boolean interpretation flag to JSON. The script and all initial outputs are preserved in cpu_summary_attempt1/. The only correction converts that flag to a Python bool. A second CPU attempt completed successfully; formal GPU errors and retries remain zero. CPU_SUMMARY_SERIALIZATION_RECOVERY.json records both script hashes.
