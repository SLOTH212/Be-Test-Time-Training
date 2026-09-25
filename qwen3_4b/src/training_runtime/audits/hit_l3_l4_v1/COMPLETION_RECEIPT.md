# HIT 2-GPU FSDP2 Debug / Validation V1 — Completion Receipt

Overall: **PASS**

| Test | Result | Evidence | Notes |
|---|---|---|---|
| Remote root / frozen package | PASS | `FROZEN_PACKAGE_AUDIT.json` | 101 actual files; 100 listed hashes plus manifest self; zero mismatch |
| GPU allowlist / NCCL | PASS | `GPU_VISIBILITY_AUDIT.json; L3_NCCL_REPORT.json` | Physical 2,3 only; logical cuda:0,1 |
| Model / L30 / FSDP2 | PASS | `MODEL_CONFIG_AUDIT.json; TTT_LAYER_COVERAGE.json; FSDP2_RUNTIME_AUDIT.json` | 36/36 blocks plus root; L30 runtime gradient-role audit |
| 32K forward | PASS | `L3_32K_FORWARD_REPORT.json; CHUNK_TRACE.json` | 8 chunks of 4096; six TTT layers |
| 32K train step | PASS | `L3_32K_TRAIN_STEP_REPORT.json` | One synthetic AdamW step; finite loss/grad/weights |
| DCP save / resume | PASS | `DCP_SAVE_REPORT.json; L4_RESUME_PARITY.json` | Model/optimizer/scheduler/RNG/cursor/tokens validated |
| World-size fail closed | PASS | `WORLD_SIZE_FAIL_CLOSED_REPORT.json` | Rejected before model build/training |
| Failure propagation | PASS | `FAILURE_PROPAGATION_REPORT.json` | Rank1 did not progress after rank0 failure |

| GPU | Physical ID | Logical ID | Peak allocated | Used? |
|---|---:|---:|---:|---|
| NVIDIA RTX A6000 | 2 | 0 | 20690839040 bytes | Yes |
| NVIDIA RTX A6000 | 3 | 1 | 20690839040 bytes | Yes |

L4 numeric note: model max abs 1.14440918e-05, model max relative RMS 3.25019506e-06, optimizer global relative RMS 1.55867639e-05; within repeated-resume control max abs 0.000366210938.

REAL_FORMAL_TRAINING_EXECUTED=false

SAFE_TO_PROCEED_WITH_REAL_DATA_INTEGRATION=YES
