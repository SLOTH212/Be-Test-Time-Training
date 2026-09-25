# Llama 3.1 NTP port v1

This is a correctness-validated model port, with tiny CPU tests. It is not a
formal Llama-3.1-8B run or a validated 8B/32K distributed configuration.

The model-family skeleton is the exact local In-Place-TTT commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1`. Training and inference attention,
rotary embeddings, normalization, cache position plumbing, and state-dict names
come from its separate Llama implementations. NTP arithmetic, contextual target,
projection, zero-initialized per-channel gate, state transition, and lifecycle
come from the completed Qwen3-0.6B frozen source, not the old Llama objective.

Run from this directory with Python 3.11 and the recorded environment:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
/home/USER/conda_envs/ruler_generation_v1/bin/python tests/run_tests.py
```

For an extracted package, append `--extracted`; this verifies bundled references
without depending on their original absolute paths. No network or weights are
needed. Tests use temporary directories for tiny model roundtrips and remove them.

Family selection for a common runtime:

```python
from adapters.model_family import build
model, config = build(
    model_path, model_family="llama", ttt_layers=[0, 6, 12, 18, 24, 30],
    chunk_size=4096, device="cuda",
)
```

Add this directory and `source/` to `PYTHONPATH`. `build` returns the existing
`(model, config)` interface, uses local assets only, BF16 and SDPA, and enables
non-reentrant checkpointing for training. `adapters.pipeline.bind_family_builder`
provides reversible injection into a common caller of `formal.build`. No
optimizer, scheduler, dataloader, checkpoint, accumulation, or distributed
implementation is forked. The frozen run-specific Qwen launcher itself still has
Qwen-scale and data-identity guards; it is not a Llama launch command. Future
formal Llama launch configuration must supply validated Llama identities through
the existing common runtime. No existing launcher or runtime was edited.

`model_family="qwen3"` selects byte-identical frozen Qwen snapshots and retains
their original 1024 chunk guard. The Llama copy of the common core changes only
that guard to a positive configurable chunk size. This is explicitly a candidate
scale-up parameterization; the update rule is unchanged.

Inference uses `inference=True`. Use `adapters.inference.fresh_cache(model)` for
sample reset and `action_scope(model, action)` for OFF / configured layer / ALL.
Reset discards KV and TTT state together. A fresh `generate` call creates a new
cache. Discard the old cache when changing independent samples or action replays.
The unchanged Dynamic instrumentation can attach to the Llama MLP interface;
only static installation was tested. No Fixed or Dynamic benchmark ran.

Training requires original 2-D validity masks and explicit document boundaries.
The boundary contract resets NTP fast weights; it does not add new document-level
attention isolation. Training computes a last partial-chunk update, which is
then discarded at the document boundary. Inference updates complete prompt chunks
only; the partial tail is forwarded using the current fast weight. The frozen
single-token continuation detector suppresses all continuation updates. Therefore
TTT-enabled token-at-a-time prompt streaming is not equivalent to full prefill:
single-token calls after cache position zero are treated as generation, exactly
as in the Qwen authority. OFF incremental decoding and mathematically comparable
TTT multi-token prompt blocks are tested separately.

At exact gate zero, the projection gradient is zero, while the gate gets a
gradient. After the gate opens, projection gradients flow. The retained Conv
parameter is unused and has no gradient. All base parameters remain trainable,
as in Stage1. The port rejects biasful MLP configurations when TTT is enabled;
Llama-3.1 homologous fast weights are bias-free.

The Stage2 test executes the exact frozen answer-mask function with a test-only
CPU linear/cross-entropy substitute for its CUDA Liger kernel. Masks, shifts,
separate denominators, and answer/context weights are unchanged. This establishes
model-interface and loss-semantics compatibility; it does not validate the CUDA
kernel or FSDP2. The 32K test passes 32768 real hidden vectors through the adapted
MLP and chunk loop (eight 4096-token chunks); it does not run quadratic 32K
attention or allocate an 8B model.

Before real Llama-3.1-8B testing: supply exact local model/config/tokenizer assets,
verify layer dimensions and Llama-3.1 RoPE, independently tokenize Stage1/Stage2
data, validate 32K capacity and the unchanged distributed runtime on GPUs, and
freeze a formal configuration. No HIT access or transfer occurred in this goal.
