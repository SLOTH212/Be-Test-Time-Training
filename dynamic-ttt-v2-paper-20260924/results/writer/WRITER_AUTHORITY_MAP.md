# WRITER AUTHORITY MAP

Authority gate: PASS. Created before numerical merging. No competing completed latest authority found: r2_mechanism_cpu_20260923 contains NOT_RUN.md; latest_dynamic_mechanism_cpu_20260923/completion is a supplementary numerical analysis of the same source cohort, not a competing experiment. Earlier Stage-1/Stage-2 historical-winner replay and 2K/cross-chunk runs are not final re-search authorities.

Common benchmark: RULER nominal 16K, complete prompts without truncation, chunk 1024; actions OFF/L0/L6/L12/L18/L24/ALL; trained old 1.7B checkpoint.
Checkpoint weights SHA256: ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f
Checkpoint historical identity: f4e264ed6f5671f123d6c8a27c69bce3af5ffcba518c9e298398314c6147d9fd

Final writer table authority is raw fixed cells + completed current V2 selected Dynamic winners, not manuscript values.

## A Fixed7
- absolute_path: `/home/zonghan/ttt/runs/formal/fixed7_old1p7b_chunk1024_v2_reuse_20260918`
- completion_status: existing `/home/zonghan/ttt/runs/formal/fixed7_old1p7b_chunk1024_v2_reuse_20260918/COMPLETE.json`; SHA256 `0956b6c59721b50b31c02531e2ad2374c5e64bd19be6856b5364902df051cf32`
- completion_evidence: `{"at": "2026-09-18T13:01:02+08:00", "state": "COMPLETE", "total_cells": 45500}`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/runs/formal/fixed7_old1p7b_chunk1024_v2_reuse_20260918/source/executor.py`
- source_sha: `37411ae15cdfbffbc9852839092c2715cd36e45085aee3569549576b5221eb69`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 6500 samples / 45500 cells
- why_authoritative: 1435 reused V2 samples plus 5065 completed V2 missing samples; fixed-source hash frozen by Dynamic protocol.

## B Dynamic local
- absolute_path: `/home/zonghan/ttt/runs/formal/dynamic_old1k_v2_noncwe_local_20260918`
- completion_status: existing `/home/zonghan/ttt/runs/formal/dynamic_old1k_v2_noncwe_local_20260918/COMPLETE.json`; SHA256 `a96ab209616a3b2a70fe5944f970f37a30ff59ff5bc839b8504a942e41a08209`
- completion_evidence: `{"status": "COMPLETE", "n": 717, "machine": "local", "protocol_hash": "b004b78c600d12f2660cbc2b92e6d252f019df941b509c644d271b8ffebee50a", "at": "2026-09-20T22:31:40.808480+08:00"}`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918/runtime/executor.py`
- source_sha: `85a659a6ab2d5c97187cd464db08e8074722fb7d4567284f04cb17671fd5f812`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 717 assigned searched samples
- why_authoritative: Completed frozen local partition.

## B Dynamic remote
- absolute_path: `/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921/dynamic_old1k_v2_remote_transfer_20260921/dynamic_old1k_v2_remote_results`
- completion_status: existing `/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921/dynamic_old1k_v2_remote_transfer_20260921/dynamic_old1k_v2_remote_results/COMPLETE.json`; SHA256 `da28057a8c993d6c7bafb267a5b93c6aaa5f50a9850d58f34b535ccbaaea7611`
- completion_evidence: `{"status": "COMPLETE", "n": 716, "machine": "remote", "protocol_hash": "b004b78c600d12f2660cbc2b92e6d252f019df941b509c644d271b8ffebee50a", "at": "2026-09-21T07:17:34.064279+08:00"}`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918/runtime/executor.py`
- source_sha: `85a659a6ab2d5c97187cd464db08e8074722fb7d4567284f04cb17671fd5f812`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 716 assigned searched samples
- why_authoritative: Completed complementary partition; same protocol/model/executor and transfer checksum manifest.

## B Dynamic statistics
- absolute_path: `/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921`
- completion_status: existing `/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921/FINAL_AUDIT.json`; SHA256 `8a94cefc1ad3be1f36a3d9a44c51197e304d2246b7a1d6481c7f13030d6ad808`
- completion_evidence: `{"status": "PASS", "transfer_files_verified": 740, "protocol_sha256": "b004b78c600d12f2660cbc2b92e6d252f019df941b509c644d271b8ffebee50a", "full6000": {"n": 6000, "sample_best": 84.`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921/audit_merge.py`
- source_sha: `1bb21abb4f9d5a9ee8316552f7bf814af649dfe18f732eba6d2b87b276225b25`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 12 tasks, all non-CWE parent IDs
- why_authoritative: Current merge verifies both raw partitions; defines ordering and bootstrap.

## C mechanism, P omitted
- absolute_path: `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921`
- completion_status: existing `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921/COMPLETE.json`; SHA256 `28a3d289faaa9ab12bcde357d964244901b462a6d5fd5fa7a451f48de062d873`
- completion_evidence: `{"at": "2026-09-22T13:44:19+08:00", "state": "COMPLETE"}`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921/runtime/executor.py`
- source_sha: `b1bd0679ad35bd236f693f80b17c29ddb598920e262fa3ecabb597da2c4e6abf`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 170 improved; 165 reset/K;157 nontrivial reset/deletion
- why_authoritative: Protocol binds exact latest Dynamic source hash and improved cohort.

## D exact R2
- absolute_path: `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921/r2`
- completion_status: existing `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921/r2/../COMPLETE.json`; SHA256 `28a3d289faaa9ab12bcde357d964244901b462a6d5fd5fa7a451f48de062d873`
- completion_evidence: `{"at": "2026-09-22T13:44:19+08:00", "state": "COMPLETE"}`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921/run.py`
- source_sha: `61ee36219738794e199ef5f2da8b94062b28fb969494c36ffcc811a098bbdd8e`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 170; full 42*(T-1) candidates per sample
- why_authoritative: Same current mechanism protocol; no V1 candidates.

## E CPU R2 structure
- absolute_path: `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923`
- completion_status: existing `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923/integrity.json`; SHA256 `13d5ac2b417ec48bc4020e33ec544870d8983ba9879035b74d07099cb1873352`
- completion_evidence: `{"status": "PASS", "protocol_sha256": "b004b78c600d12f2660cbc2b92e6d252f019df941b509c644d271b8ffebee50a", "sources_verified": 1433, "candidate_records_checked": 531113, "machine_co`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923/audit.py`
- source_sha: `e061b282975a782536e02dfa1cbfa4b1ce2a3ba357e025b1dbe6f0bac1d4daef`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 170 current improved; full R2; supplementary set-wise statistics
- why_authoritative: Verifies all current Dynamic/R2 sources and complete candidate coverage.

## E supplementary structure
- absolute_path: `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923/completion`
- completion_status: existing `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923/completion/FINAL_STATUS.json`; SHA256 `1f44174a3262a16b4af2c81b0cff4a91f6a20c9e97f38950596bf8de5af3ae2b`
- completion_evidence: `{"numerical_audit": "COMPLETE", "plotting": "PAUSED_BY_USER", "complete_curve_atlas": "NOT_COMPLETED_NOT_DELIVERED", "new_model_execution": false, "CPU_only": true, "inputs_rehashe`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/audits/latest_dynamic_mechanism_cpu_20260923/completion/numerics.py`
- source_sha: `6375d11ef7bd15809cd7e2fa636476a80f005bb577317b1af6204ea2403abdb9`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: Same cohort, set-wise reversal/directions/centered ranks
- why_authoritative: Numerical audit COMPLETE; plotting paused and outside current request.

## F original V2 executor validation
- absolute_path: `/home/zonghan/ttt/audits/reference_executor_v2_stage1_20260915`
- completion_status: existing `/home/zonghan/ttt/audits/reference_executor_v2_stage1_20260915/V2_FORMAL_COMPLETION.json`; SHA256 `599fec697030eb4b276ee3cd1a201361e1887a6249236826aeb6680274469575`
- completion_evidence: `{"checkpoint_unchanged_n": 1384, "error_n": 0, "protocol_sha256": "07def65bae4f276945a78ef4bbbcffac9980bbf7e1782781e562e2b83a99b034", "retry_n": 0, "state_reset_pass_n": 1384, "sta`
- checkpoint_identity: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f` (weights; historical identity above)
- executor_version: path-unified V2; source `/home/zonghan/ttt/audits/reference_executor_v2_stage1_20260915/source/reference_executor_v2.py`
- source_sha: `8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c`
- context: nominal 16384; chunk: 1024; action_set: OFF,L0,L6,L12,L18,L24,ALL
- sample_scope: 144 precheck trajectories; 1384 formal Stage-1 trajectories
- why_authoritative: Original V2 precheck and formal validation. Hash differs from final adapted executor; limits and static lineage audit required.

## Bootstrap authority
`/home/zonghan/ttt/audits/dynamic_old1k_v2_merge_20260921/audit_merge.py`: task lexicographic; within-task sample order from `/home/zonghan/ttt/transfer/dynamic_old1k_v2_noncwe_half_20260918/data/fixed_all6500.jsonl`; seed 20260921; numpy default_rng PCG64; 20000 replicates, 1000 per batch; percentile linear .025/.975. Mechanism uses COHORT.json order and summarize.py. R2 structure uses separate recorded CPU analysis protocols, not Dynamic CI seed.

## Scope of validation
Original validated source: 8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c. Final Dynamic source: 85a659a6ab2d5c97187cd464db08e8074722fb7d4567284f04cb17671fd5f812. Fixed completion source: 37411ae15cdfbffbc9852839092c2715cd36e45085aee3569549576b5221eb69. Do not claim byte identity or full rerun of all original prechecks on adapters. Static lineage and existing bridges will be reported separately.

## Manuscript
User-supplied `/home/zonghan/.codex/attachments/d07adb75-9cd3-4d3c-b018-769fc74bc4ac/ICLR2026_Zonghan.pdf`: actual 20 pages. Used exclusively for replacement locations/old values.

All downstream outputs are independent CPU audit derivatives; no formal file will be modified.
