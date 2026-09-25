# Frozen scientific semantics

The byte-exact `dynamic/formal_dynamic_runner.py` is the Dynamic source named by the completed 1.7B formal run. The fixed evaluator lineage imports `code/inference_model/hf_qwen3/modeling_qwen3.py`; it does not substitute the training-side model implementation.

Fast weights target `mlp.down_proj.weight`. The objective is contextual-hidden direct outer-product next-token prediction without a scalar loss, with learning rate 1, inference Frobenius delta clip `1e-5`, and apply-then-update ordering. Complete prompt chunks update; the incomplete tail is retained and forwarded without update. Generation never updates TTT state. Every candidate is a clean replay, every sample resets local state, downstream computations are recomputed, and stored-delta stitching is forbidden. KV/generation state is candidate-local and not reused across candidates or samples.

Historical inference uses BF16, SDPA, greedy decoding (`do_sample=false`, `num_beams=1`, `use_cache=true`), left padding, model EOS as padding, and the benchmark row's explicit `max_new_tokens`. Temperature, top-p, and top-k are explicitly null.

Dynamic means **Dynamic Recomputed Lower Bound**. It is bounded approximate beam coordinate search, not an oracle or global optimum. Search uses all constant trajectories as seeds, beam 4, one forward sweep, and terminal RULER score. The unique-candidate replay bound is `A + A(A-1) + (C*S-1)*B*(A-1)` for `C>0`; the final trace capture is separately accounted. This is 385 for historical `A=7,C=15,B=4,S=1`, and 260 for scale-up `A=8,C=8`.

Fixed evaluation includes all 13 RULER tasks including CWE. Historical Dynamic aggregation explicitly excludes CWE and searches only non-ceiling samples; ceiling samples receive `Dynamic := Sample Best := 1` without search.
