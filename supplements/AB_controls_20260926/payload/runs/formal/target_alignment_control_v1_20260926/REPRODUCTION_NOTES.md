# Target-alignment experiment: interpretation and replay details

The experiment is independent of the historical formal outputs. No shared evaluator or model source was edited. `adapter.py` copies the historical `install_dynamic_forward` function with only the active-update call routed to an optional controller, including explicit layer/chunk/action metadata. The controller defaults disabled and directly calls the original step. All model loading, tokenization, action masks, complete-chunk loop, tail/generation handling and scorer remain the historical V2 implementation.

## What is shuffled

The model's post-attention LayerNorm hidden is z; u=SiLU(gate_proj(z))*up_proj(z). Native pairs u at0..C-2 with gated z at1..C-1. The hook permutes rows of the gated target, then evaluates the unchanged three-operand einsum. It does not permute u, change P/g, explicitly preproject targets with a different GEMM contraction, or import a different chunk's target. For every complete chunk n_targets1023, deterministic derangement has0 fixed points. Equal-valued target rows can still occur; the definition concerns positional pairing, not forced semantic inequality.

## Native references and evolving branches

Native per-write scalar raw/clipped/represented norms are measured in an actual clean Native replay and indexed by sample/layer/chunk. B2 reads these scalars as the prescribed magnitude reference. No Native future activation, target, cache or fast-weight tensor is stored in the controller's reference map. B1 and B2 compute their own later u and v as their states diverge; their raw shuffled updates will generally differ, even under the same local permutation. Consequently the Parquet distinguishes `shuffle_raw_norm_B1` from `shuffle_raw_norm_B2`.

Fresh generation cache and clean checkpoint-based fast-weight initialization apply to every branch. Native replay/prediction is checked for every formal sample, then repeated after its corrupted branches. Identity and disabled Native comparisons are smoke-only validation; explicit Native original-step tensor equality verifies the copied arithmetic there. Formal Native update hashes are retained. Formal corrupted branches preserve write norms/metadata/permutation hashes rather than copying every huge weight tensor to CPU for hashing. This coverage is not a claim of full internal trace equality on corrupted paths.

## B2 limitation under unchanged clipping

B2 computes BF16(raw_shuffle*alpha), original FP32 norm and clipping factor cast back to BF16, BF16 clipped update, and BF16 state addition. The actual represented difference is then measured in FP32. Clipping saturation and quantization mean the target norm may not be attained by the fixed finite scalar search. The nearest *tested* candidate is retained; neither global mathematical optimality nor universal <=1% matching is asserted.

`CHECK_B11` protocol correctness and strict amplitude attainment must be read separately. A successful execution/status does not establish a successful equal-magnitude scientific control. If many writes exceed1%, Native-vs-B2 score differences retain a magnitude confound. B1 reports the natural total effect of shuffling target pairing, which includes changes in update magnitude.

Zero Native represented norm uses alpha0, preserving represented-zero change through the original pipeline. An active schedule event remains active even if its actual numerical displacement is0. Norm diagnostic counts are write-level; inference/statistics remain example-level. Never treat18800writes or850seed-results as independent examples.

## Frozen randomness and aggregation

Five seeds20260926..20260930. A local CPU RNG derives each permutation from SHA256 of algorithm/sample/layer/chunk/seed/n_targets. B1/B2 share the permutation coordinates, not targets or hidden states. Random construction and scalar calibration never accept a reference answer, prediction or reward. Five seeds average within each of170examples. Recovery is not clipped; ratio of summed gains is distinct from average per-example recovery. Task-stratified bootstrap uses20000 draws andseed20260921 in the historical order.

Source hashes were frozen before GPU execution. The historical tokenizer advisory is preserved without changing tokenizer settings; stored Native prediction/score parity gates the run. PyArrow21.0.0 was installed only in this experiment's `deps/` for requested Parquet export; the existing runtime environment was not modified.

Main entrypoint: `pipeline.py`. Required inputs live in the read-only historical mechanism directory and model checkpoint referenced in config; this directory alone does not bundle weights/data/runtime. Existing completed sample records may be resumed only under matching configuration and source hashes. `trajectory_audit.jsonl` is frozen input provenance; final runtime checks are written separately to `trajectory_result_audit.jsonl`.
