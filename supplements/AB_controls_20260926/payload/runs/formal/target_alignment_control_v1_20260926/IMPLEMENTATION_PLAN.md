# Within-Chunk Target-Alignment Control — audit and frozen implementation plan

Authority ../mechanism_old1k_v2_20260921: COHORT170, exact Native winner sequence per sample; 3760 selected layer/chunk writes per pass, 18800 writes per5-seed condition. Native/SB reference from frozen records, no new search or selection. Actions OFF,L0,L6,L12,L18,L24,ALL; candidates0,6,12,18,24; chunk1024; same trained1.7B checkpoint/hash, BF16 SDPA, LR1, clip1e-5, original tokenizer/scorer/greedy limits.

## Source audit before code changes

runtime/code/hf_models/hf_qwen3/ttt_state_core.py: target_source[:,1:] paired with hidden[:,:-1]; direct gate vector multiplication; original einsum bti,btd,de->ei multiplied by learning rate; FP32 norm, clipping scale cast to BF16 before BF16 multiply; BF16 add to current weight. Output F.linear executes before the update. runtime/code/inference_model/hf_qwen3/modeling_qwen3.py:357–360 defines z as post-attention LayerNorm hidden. runtime/executor.py:87 defines u as SiLU(gate_proj(z))*up_proj(z); complete-chunk-only update dispatch, frozen action mask, tail/generation read-only. Algebra matches v=P^T(g*z_next); no mandatory discrepancy. Preserve three-operand einsum and operation order; do not substitute algebraically equivalent GEMM order.

## Hook design

Independent runtime adapter generated from the existing install_dynamic_forward function with one changed update call routed through controller.step(layer,chunk,action). No historical/shared files changed. Controller defaults disabled and delegates to original ttt_chunk_step. Enabled Native reconstructs exact original operations and records raw/clipped/represented norms and hashes. Identity uses explicit identity target permutation and must reproduce Native update/state/output hashes and prediction. B1/B2 permute gated target rows before original einsum. All branch u and target tensors come from that branch's own evolving forward. Native future tensors never enter a corrupted branch. B2 reads only corresponding Native scalar write norms and diagnostic hashes, never its tensors. Every branch full clean replay, fresh default cache, native sequence unchanged.

## Frozen randomization

Seeds [20260926,20260927,20260928,20260929,20260930], shared B1/B2 permutation at each sample/layer/chunk/seed. Stable SHA256-derived local CPU RNG; rejection-sample random permutations until derangement, max128 tries, then cyclic shift1 fallback. Effective valid target indices only; count1023 for full unpadded1024 chunks. n<=1 recorded explicitly, no invented cross-boundary target. Save permutation hash/count/fixedpoints; all actual n>1 must have0fixedpoints. No reward/reference input.

## B2 scalar matching

Reference target is actual represented norm of corresponding write in true Native replay, not a hypothetical unshuffled update in the corrupted state. For each alpha evaluate raw_shuffle*alpha in execution BF16, exact original FP32-norm clipping with BF16 factor/multiply, BF16 add to CURRENT B2 weight; FP32 actual difference norm. Never post-clip scale or adjust direction. Native zero -> alpha0 and exact zero. Fixed bounded search candidates0,1, alpha_clip*2^j for j=-16..16; then12 rounds log-midpoint refinement on immediate tested neighbors of current best. Closest evaluated error, ties smalleralpha. Nonfinite candidates invalid. No monotonicity/globaloptimum claim. >1% closest solutions retained per user; report all errors and attainable-range observations. This can fail strict magnitude control even when execution is correct: if strict1%not attained, never claim allwritesmagnitude matched. Final mechanism interpretation must explicitly qualify matching failures; B2 significance alone insufficient to assert a rigorous equal-magnitude effect if failures substantial.

## Validation and execution order

Source/cohort/config freeze -> disabled Native and enabled Native exact update recovery -> explicit identity test -> first5 frozen cohort samples smoke (includes OFF/single/ALL action regimes), all5 seeds forB1+B2 and seed0 repeats -> smoke execution gate ->170 formal, two workers CPU8–15/16–23 ->sample aggregation/bootstrap ->finalaudit. Formal Native checked against stored reward/prediction every sample. Native/identity update hashes checked at smoke; disabled_native comparison included. Source integrity checked before/after execution. No extra clipping/seed/cohort changes based on outcomes. Normclosest exceptions do not silently become exact matches.

## Statistics

Sample is unit; average5 seeds separatelyB1/B2. Native−B1, Native−B2, B2−SB; sample recovery unbounded and aggregate ratio distinct. Task-stratified paired bootstrap20000 seed20260921, sorted task order, frozen cohort order, batches1000, percentilequantiles. Equality tol1e-12 score units. Task means descriptive. Reports distinguish execution integrity and strict B2 magnitude-match attainment. Requested write diagnostics parquet plus raw sample JSON, seed/example/task tables and machine-readable final contract.
