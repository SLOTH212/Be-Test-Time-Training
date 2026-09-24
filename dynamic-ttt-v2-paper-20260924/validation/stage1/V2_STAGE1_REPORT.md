# Reference executor V2: Stage-1 core robustness revalidation

STRONG_RETENTION

Of173 historical Dynamic-improved samples, 153 retain strict superiority over all seven newly replayed V2 constants, 19 tie, and 1 fall below the new SampleBest. Success retention is88.439306%. Mean gap is44.874759152pp, with descriptive95% percentile CI[39.72061657032756, 50.105973025048165].

THIS CI DOES NOT ESTIMATE THE FULL 6000-SAMPLE EFFECT.

This is a success-conditioned historical-winner robustness study. No new Dynamic search was performed, and these scores are not a V2 Dynamic LB. The newly evaluated historical Native is compared only against the same sample's seven new V2 constants. No old score enters SampleBest_V2 or gap_V2. The old scores remain historical observations under the old executor.

## Executor and validation

All five candidate layers always use the same functional complete-chunk, tail and generation path. Action labels only mask writes. Inactive layers retain effective weights; every trajectory starts from checkpoint weights and a fresh cache. The original update rule/scorer/greedy budget/BF16/SDPA and non-candidate-layer semantics are unchanged. Historical executor files are not edited. See REFERENCE_EXECUTOR_V2_DESIGN.md, V2_SOURCE_DIFF.patch and V2_SOURCE_SHA256.json.

Precheck used the previous formal32: 4 same-Native repeat pairs; 64 zero-write comparisons covering all32 with Z_OFF/Z_NATIVE/Z_ALL; 15 common-prefix pairs; 5 apply-then-update cases, one per candidate layer. All required comparisons were PASS_EXACT. The4-run uninstrumented/instrumented bridge matched prediction hashes and rewards4/4. Precheck trajectories (144) and bridge trajectories (4) are separate from formal1384.

## Core result

| Metric | Value |
|---|---:|
| SB_V2 mean (%) | 42.215799614643544 |
| NativeOld_V2 mean (%) | 87.09055876685935 |
| Mean gap (pp) | 44.874759152215795 |
| Median gap (pp) | 33.333333333333336 |
| Minimum gap (pp) | -33.333333333333336 |
| Maximum gap (pp) | 100.0 |
| GT | 153 |
| EQ | 19 |
| LT | 1 |

Tie tolerance is1e-12 on raw scores. Positive/zero/negative raw gap counts are (153, 19, 1). The paired sample bootstrap uses20000 resamples, NumPy PCG64 seed20260915 and linear percentile quantiles.

## Historical score shifts

| Action | N | Mean pp | Median pp | Min pp | Max pp | Prediction changes | Reward changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | 173 | -0.3853564547206165 | 0.0 | -66.66666666666666 | 0.0 | 4 | 1 |
| L0 | 173 | 0.0 | 0.0 | 0.0 | 0.0 | 2 | 0 |
| L6 | 173 | 1.5414258188824663 | 0.0 | 0.0 | 100.0 | 6 | 5 |
| L12 | 173 | 0.3853564547206165 | 0.0 | 0.0 | 66.66666666666666 | 6 | 1 |
| L18 | 173 | 0.0 | 0.0 | 0.0 | 0.0 | 4 | 0 |
| L24 | 173 | 0.3853564547206166 | 0.0 | -66.66666666666666 | 100.0 | 7 | 3 |
| ALL | 173 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| NATIVE | 173 | -6.165703275529865 | 0.0 | -100.0 | 0.0 | 16 | 16 |

Fixed score-change sample count=6 means unique samples with at least one changed constant score; changed fixed cells=10. Fixed prediction changes count cells out of1211. Native changes count samples out of173. Score changes use exact raw-float inequality; tolerance-based counts are also in the CSV.

## Task-wise conditional robustness

| Task | N | GT | EQ | LT | Mean gap pp | Historical mean gap pp | Native reward changes | SB reward changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ruler_fwe_16k | 24 | 14 | 9 | 1 | 18.055555555555557 | 33.333333333333336 | 8 | 3 |
| ruler_niah_multikey_1_16k | 1 | 1 | 0 | 0 | 100.0 | 100.0 | 0 | 0 |
| ruler_niah_multikey_2_16k | 4 | 4 | 0 | 0 | 100.0 | 100.0 | 0 | 0 |
| ruler_niah_multikey_3_16k | 14 | 12 | 2 | 0 | 85.71428571428571 | 100.0 | 2 | 0 |
| ruler_niah_multiquery_16k | 12 | 12 | 0 | 0 | 27.083333333333332 | 27.083333333333332 | 0 | 0 |
| ruler_niah_multivalue_16k | 8 | 8 | 0 | 0 | 28.125 | 28.125 | 0 | 0 |
| ruler_qa_hotpotqa_16k | 18 | 10 | 8 | 0 | 55.55555555555556 | 100.0 | 6 | 2 |
| ruler_qa_squad_16k | 15 | 15 | 0 | 0 | 100.0 | 100.0 | 0 | 0 |
| ruler_vt_16k | 77 | 77 | 0 | 0 | 33.506493506493506 | 33.506493506493506 | 0 | 0 |

No significance claim is made for small task subsets.

## Decision and scope

Consider separately authorized1435 historical-winner replay / formal Dynamic re-search. The pre-frozen decision bucket is STRONG_RETENTION. These thresholds guide resource allocation and are not statistical significance criteria. No next-stage search, R2, Reverse, Reset, Deletion, K/P, CWE, or cross-model experiment was run.

A positive NativeOld_V2 gap certifies that particular previously discovered nonconstant trajectory against the complete seven-constant family under this V2 executor. It does not certify search optimality, a deployable controller, or a full6000 population effect. Non-retention describes old winners under this executor and does not prove that no better V2 nonconstant trajectory exists.

## Identity, safety and artifacts

All173 historical Native sequences and hashes were preserved; all1211 fixed cells are present with no duplicates. Each of1384 formal runs verifies the input/token/mask hashes, budget, action identity, and unchanged writable checkpoint digests. New evaluator/cache creation and explicit per-trajectory cleanup prevent carried fast-weight state. Errors and retries, if any, remain separately recorded; none are silently discarded. Only hashes and scores are saved, without raw prompt/reference/prediction text.

## Required output

```text
REFERENCE_EXECUTOR_V2_STAGE1_COMPLETE

[V2_AUTHORITY]
CHECKPOINT_MATCH=YES
MODEL_HASH_MATCH=YES
SCORER_MATCH=YES
TOKENIZER_MATCH=YES
CONTEXT=16384
CHUNK=1024
DTYPE=bfloat16
GPU=Single RTX5090
V2_SOURCE_SHA256=8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c
V2_PROTOCOL_SHA256=07def65bae4f276945a78ef4bbbcffac9980bbf7e1782781e562e2b83a99b034

[V2_PRECHECK]
FORMAL32_N=32
SAME_PATH_REPEAT_STATUS=PASS_EXACT
ZERO_WRITE_SCHEDULE_INVARIANCE_STATUS=PASS_EXACT
FUTURE_ACTION_PREFIX_STATUS=PASS_EXACT
APPLY_THEN_UPDATE_STATUS=PASS_EXACT
CHECKPOINT_ISOLATION_STATUS=PASS_EXACT
V2_PRECHECK_STATUS=PASS_EXACT

[V2_FORMAL_REPLAY]
SAMPLE_N=173
TRAJECTORY_N=1384
COMPLETED_N=1384
ERROR_N=0
CHECKPOINT_UNCHANGED_N=1384
STATE_RESET_PASS_N=1384

[V2_FIXED7]
EXPECTED_CELL_N=1211
OBSERVED_CELL_N=1211
MISSING_N=0
DUPLICATE_N=0

[V2_CORE_RESULT]
SB_V2_MEAN=42.215799614643544
NATIVE_V2_MEAN=87.09055876685935
NATIVE_MINUS_SB_V2_PP=44.874759152215795
NATIVE_MINUS_SB_V2_CI=[39.72061657032756, 50.105973025048165]
GT_N=153
EQ_N=19
LT_N=1
SUCCESS_RETENTION_RATE=0.884393063583815

[V2_HISTORICAL_SHIFT]
FIXED_SCORE_CHANGE_SAMPLE_N=6
NATIVE_SCORE_CHANGE_SAMPLE_N=16
FIXED_PREDICTION_CHANGE_N=29
NATIVE_PREDICTION_CHANGE_N=16
MEAN_NATIVE_SCORE_SHIFT_PP=-6.165703275529865
MEAN_SB_SCORE_SHIFT_PP=1.7341040462427746

[V2_TASKS]
TASK_N=9
TASKS_WITH_GT=9
TASKS_WITH_EQ=3
TASKS_WITH_LT=1

[SCIENTIFIC_INTERPRETATION]
PATH_UNIFIED_EXECUTOR_VALID=YES
HISTORICAL_SEQUENTIAL_UTILITY_RETENTION=153/173 = 88.439306%
CAN_CLAIM_V2_DYNAMIC_LB=NO
CAN_CLAIM_HISTORICAL_WINNER_ROBUSTNESS=YES
FULL_DYNAMIC_RESEARCH_NEEDED=NOT_RUN; Consider separately authorized1435 historical-winner replay / formal Dynamic re-search

[NEXT_STAGE_RECOMMENDATION]
DECISION_BUCKET=STRONG_RETENTION
RECOMMENDED_NEXT_STEP=Consider separately authorized1435 historical-winner replay / formal Dynamic re-search
```
