TARGET_ALIGNMENT_CONTROL_STATUS=PASS
COHORT_N=170
COHORT_EXACT=true
TRAJECTORY_EXACT=true
N_SEEDS=5
NATIVE_REPLAY_PASS=true
B1_MEAN=46.935294117647054
B2_MEAN=46.984313725490196
NATIVE_MEAN=93.52941176470588
SAMPLE_BEST_MEAN=40.411764705882355
NATIVE_MINUS_B1_PP=46.59411764705883
NATIVE_MINUS_B1_CI95=[43.77053921568627, 49.315735294117644]
NATIVE_MINUS_B2_PP=46.545098039215695
NATIVE_MINUS_B2_CI95=[43.81960784313726, 49.13725490196078]
B2_MINUS_SAMPLEBEST_PP=6.572549019607843
B2_MINUS_SAMPLEBEST_CI95=[4.147009803921569, 9.176470588235293]
B2_MEAN_GAIN_RECOVERY=0.11058823529411765
B2_AGG_GAIN_RECOVERY=0.12373569582871882
B2_MAG_MATCH_MEDIAN_REL_ERROR=0.012849997235259647
B2_MAG_MATCH_MAX_REL_ERROR=0.711000622226491
B2_MATCH_WITHIN_1PCT_RATE=0.4570212765957447
B2_STRICT_MAGNITUDE_MATCH=false
FINAL_INTEGRITY=PASS

# Within-Chunk Target-Alignment Control

Scores and differences above are percentage points/percent; recovery ratios are unbounded unitless ratios.
All 170 frozen improved examples are retained. No trajectory search or outcome-based selection. Five seeds are averaged within each example before paired task-stratified bootstrap.

| Condition | Mean score (%) |
|---|---:|
| sample_best | 40.41176471 |
| native_score | 93.52941176 |
| B1 | 46.93529412 |
| B2 | 46.98431373 |

## Magnitude control and interpretation

B2 strict matching rate: 45.702128%; 10208 writes exceed 1%. The scalar search retains the closest tested solution under unchanged clipping and BF16 addition.
If matching exceptions are substantial, these results do not establish an equal-magnitude causal contrast. A significant Native–B2 difference alone cannot remove the residual magnitude confound.
No global optimizer guarantee is claimed; finite deterministic search, clipping saturation and BF16 quantization are disclosed.

## Provenance and limitations

Frozen configuration SHA256: `72455e3e1d84a2db927078cae98e9a213e3eb71dc0eb418bb9b4ff26ea9b610a`.
Historical source: `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921`; source integrity status: PASS.
Identity recovery and seed determinism are validated on the frozen smoke cohort; Native replay and stored prediction/reward agreement are checked for every formal sample.
Raw result JSON retains write hashes and per-seed diagnostics. Parquet contains paired B1/B2 write rows; CSV example data retain unclipped recovery. Task results are descriptive.
Protocol deviations: no unrecorded deviation is inferred by this summarizer. See configuration, smoke record and source integrity evidence for execution provenance; norm-tolerance exceptions are disclosed above.

Sample-level comparisons: {"B2_above_native": 0, "B2_above_sample_best": 52, "B2_at_or_below_sample_best": 118, "B2_below_native": 161, "B2_equal_native": 9}

Execution errors: []
