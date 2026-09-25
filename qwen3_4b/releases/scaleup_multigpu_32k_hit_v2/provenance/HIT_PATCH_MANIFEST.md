# HIT Patch Manifest V2

Complete byte-oriented unified diff: HIT_PATCHSET.diff.

| Patch | Classification | Files | Root cause | Validation | Scientific semantics changed? |
|---|---|---|---|---|---|
| HIT-P1 FSDP2 hidden-forward handling | CORRECTNESS_CRITICAL | code/lib/fsdp2_runtime.py<br>code/workers/distributed_train_worker.py | Inner-decoder calls bypassed root FSDP2 hooks, producing mixed Tensor and DTensor execution. | audits/hit_l3_l4_v1/L3_32K_FORWARD_REPORT.json<br>audits/hit_l3_l4_v1/L3_32K_TRAIN_STEP_REPORT.json | No |
| HIT-P2 Stage1 document-boundary handling | CORRECTNESS_CRITICAL | code/workers/distributed_train_worker.py | Declared Stage1 document boundaries were not propagated into gated NTP. | audits/hit_l3_l4_v1/L3_32K_FORWARD_REPORT.json<br>audits/hit_l3_l4_v1/L3_32K_TRAIN_STEP_REPORT.json | No |
| HIT-P3 DCP metadata preflight | RUNTIME_CORRECTNESS | code/lib/distributed_checkpoint.py<br>code/workers/distributed_train_worker.py | Resume compatibility checks occurred after model construction. | audits/hit_l3_l4_v1/WORLD_SIZE_FAIL_CLOSED_REPORT.json | No |
| HIT-P4 Lazy optimizer-state restore | CORRECTNESS_CRITICAL | code/lib/distributed_checkpoint.py | Strict DCP load rejected legitimately absent lazy optimizer state. | audits/hit_l3_l4_v1/DCP_SAVE_REPORT.json<br>audits/hit_l3_l4_v1/L4_RESUME_PARITY.json | No |
| HIT-P5 NCCL device binding | RUNTIME_CORRECTNESS | code/lib/distributed_runtime.py | Process group and barriers lacked explicit logical CUDA device binding. | audits/hit_l3_l4_v1/L3_NCCL_REPORT.json | No |
| HIT-P6 Distributed exit lifecycle | RUNTIME_CORRECTNESS | code/lib/distributed_runtime.py<br>code/workers/distributed_train_worker.py | Collectives were unbounded and final lifecycle lacked explicit rank markers. | audits/hit_l3_l4_v1/FAILURE_PROPAGATION_REPORT.json<br>audits/hit_l3_l4_v1/COMPLETION_RECEIPT.json | No |
| HIT-P7 Runtime audit correctness | AUDIT_ONLY | code/workers/distributed_train_worker.py<br>code/tests/hit_dcp_numeric_compare.py<br>code/tests/hit_dcp_state_compare.py<br>code/tests/hit_failure_propagation.py<br>code/tests/hit_forward_32k.py<br>code/tests/hit_model_build_audit.py<br>code/tests/hit_nccl_smoke.py | Per-rank loss, memory, L30 gradient and optimizer, parity, and lifecycle evidence was incomplete. | audits/hit_l3_l4_v1/TTT_LAYER_COVERAGE.json<br>audits/hit_l3_l4_v1/L3_32K_TRAIN_STEP_REPORT.json<br>audits/hit_l3_l4_v1/L4_RESUME_PARITY.json | No |

Scale-up debug configuration is recorded separately. FORMAL_CAPACITY_CONFIG_FROZEN=false.
