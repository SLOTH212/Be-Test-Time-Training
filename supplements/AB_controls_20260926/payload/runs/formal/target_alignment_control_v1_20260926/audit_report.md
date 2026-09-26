# Target alignment execution audit

Execution status: PASS

| Check | Result |
|---|---|
| CHECK_B1_COHORT_EXACT | PASS |
| CHECK_B2_TRAJECTORY_EXACT | PASS |
| CHECK_B3_NATIVE_REPLAY | PASS |
| CHECK_B4_NO_CROSS_CHUNK_TARGET | PASS |
| CHECK_B5_NO_CROSS_LAYER_TARGET | PASS |
| CHECK_B6_NO_CROSS_SAMPLE_TARGET | PASS |
| CHECK_B7_NO_FUTURE_NATIVE_CACHE | PASS |
| CHECK_B8_FIXED_POINT_AUDIT | PASS |
| CHECK_B9_ACTION_MASK_EXACT | PASS |
| CHECK_B10_UPDATE_TIMING | PASS |
| CHECK_B11_B2_MAGNITUDE_MATCH_PROTOCOL | PASS |
| CHECK_B12_SEED_DETERMINISM | PASS |
| CHECK_B13_NO_REWARD_LEAKAGE | PASS |
| CHECK_B14_NATIVE_TARGET_RECOVERY | PASS |

Native writes: 3760; per-condition writes: {'B1': 18800, 'B2': 18800}.
Strict 1% attainment: 45.702128%; exceptions: 10208; max relative error: 0.711000622.
B4/B5/B7/B10/B13 combine frozen implementation/source integrity with recorded per-sample assertions; scalar summaries alone cannot establish execution semantics.
B11 PASS denotes the allowed nearest-tested-scalar calibration protocol, not universal 1% attainment or proof of a global optimum.

Errors:
