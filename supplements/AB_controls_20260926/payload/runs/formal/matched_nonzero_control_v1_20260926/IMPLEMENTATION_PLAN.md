# Magnitude-Matched Nonzero Control — pre-execution audit and plan

2026-09-26. Historical authority: ../mechanism_old1k_v2_20260921. All new files stay in this independent directory. Historical runtime/results remain read-only.

## Phase 1 audit findings

Frozen COHORT.json contains 170 improved samples, 165 structurally eligible, 157 nontrivial-prefix. Formal deletion has 349 target branches: 109 single-layer samples and 48 ALL-prefix samples with five branches. All 349 stored deletion target before/after hashes differ (no known zero displacement). Existing run.py:core_one uses native full action sequence, tau at last switch, reset only target layer, full clean replay; runtime/executor.py supplies existing optional inject map before reading chunk tau+1. All 1,745 recorded deletion boundary events match Native before hashes and non-target after hashes. Prefix is causal and must be audited exact for every new branch, not replaced with nonnative data.

Frozen target rule: if ALL in prefix use [0,6,12,18,24]; otherwise first non-OFF action's layer. Copy exact cohort entries/target arrays; independently assert rule, never select by new scores. Structural condition first switched chunk(one-based)/T<=.25 and mode fraction a[T//2:]>=.8; prefix contains nonOFF; boundary tau is last switch index, token position tau*1024.

Historical SUMMARY.json deletion A1 read verbatim: N157, Native93.24840764331209, Deletion54.30785562632695, gap38.94055201698513 pp, CI[35.167728237791934,42.579883227176204]. Do not re-estimate or replace this A1. For A3, use stored per-sample mean of frozen deletion branches as paired reference.

## Phase 2 implementation and frozen decisions

Reuse original run.eval_branch, ex.load_model, ex.Evaluator, original scorer and inject hook. No edits to shared evaluator. Redirect helper R/ctx.run to this output directory; source path and model hashes remain frozen. Import old module with sanitized argv and PYTHONDONTWRITEBYTECODE. Instantiate native TTTDynamicCache subclass that only records/compares prefix K/V hashes; no replacement or native post-boundary cache reuse. Native replay captures existing boundary weights and prefix KV hashes. New branch runs exact native sequence with target-only inject; all five boundary before hashes and nontarget after hashes checked exact.

Freeze direction seeds [20260926,20260927,20260928,20260929,20260930]; derive per-sample/layer/seed local generator from stable SHA256. Reward-blind Gaussian FP32 direction, one scalar calibration in execution BF16, bracket/binary deterministic bounded search, retain closest examined point; report actual norm error. Target zero, actual zero or unstable injection => BLOCKED, no silent altered definition. Any <=1% failures explicitly reported; no seed reselection/cohort dropping. Norm reductions FP32. Direction module and source hashes frozen before smoke/model execution.

Smoke: first five cohort-order eligible entries (structural selection only), covering single and ALL. All branches/all5 seeds; native exact replay; deletion replay comparison for smoke (separate output only); repeat first seed in each branch to verify tensor, score and prediction determinism. Formal: same157 including smoke samples, independent per-sample atomic records, two workers CPU8–15 and16–23. Formal launches only if smoke checks pass. Repeat first seed per branch in smoke; formal additionally at least native repeat per sample. No Dynamic search; no altered trajectory, boundary, scorer, generation or prefix KV.

Statistical unit sample. Average five seeds within branch then branches within sample. A2/A3 task-stratified paired bootstrap 20k, seed20260921, task lexical order, frozen cohort order, batches1000, percentile CI. Equality tolerance1e-12 on sample-average scores. Task breakdown descriptive. CASE1 if A3 CI lower>0; CASE3 if A3 CI upper<0 (also explicitly report negative point estimates); otherwise inconclusive, not equivalence. Report matching diagnostics and every A1–A12 check, failures retained.

## Execution order

Complete code/config/source freeze -> CPU calibration checks -> five-sample GPU smoke -> smoke gate -> full157 -> integrity/source checks -> sample aggregation/bootstrap -> final report. New outputs include requested manifests, raw results, example/task tables, diagnostics, audit and final reports. An interrupted run is resumable only under identical config/source identity; no misleading PASS on partial completion.
