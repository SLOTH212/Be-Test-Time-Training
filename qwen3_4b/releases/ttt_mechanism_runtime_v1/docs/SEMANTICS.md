# Formal mechanism semantics and limits

The source authority manifest records immutable original bytes, roles, formal runs,
and configs. Reference files are evidence, not portable launchers. Runtime adapters
load only named original definitions after checking their hashes. They never run
reference main functions. Generic execution uses Qwen3 only; Llama is deferred.

Historical actions are OFF, L0, L6, L12, L18, L24, ALL. ALL is one logical action.
Scale-up templates add L30 and use 4096-token chunks at 32768 context. No 4B cohort
is frozen. Source extraction parameterizes configured layer/action globals and the
literal chunk size; the Recorder final-normalization diagnostic layer label also
uses the actual attention-layer count. Tensor update expressions are unchanged. The dependency's
formal evaluator contains two literal 1024 chunk-size sites (model configuration
and complete-chunk counting), so the mechanism adapter parameterizes those from
its checked source in memory. The frozen dependency itself is never modified.

Reverse replays the entire native action sequence in reversed temporal order from
a clean per-sample initial state. It does not reverse layer IDs or stitch deltas.

Reset eligibility calls the original screening function: first switch is indexed
by its new-action chunk, one-based, divided by full trajectory length; this must be
<= 0.25. The suffix starts at zero-based floor(N/2). Its modal action must occupy
>= 0.8, with ties resolved by configured action order. Boundary is the one-based
first chunk of the terminal equal-action run, immediately after the last switch.

Reset replaces each configured layer's local TTTState.weight with that layer's
checkpoint down_proj.weight immediately before the boundary chunk. Base parameters
are not mutated. This is within the same layer-major prompt forward, not an
instruction to rerun earlier chunks or restore a globally synchronized model state.
There is no cache invalidation or explicit K/V replacement in the reset operation.
Prefix tokens, action schedule, RNG and generation settings remain unchanged.
Downstream computations execute normally from the intervened local state. An
unused layer is implicitly at base and records a zero-residual reset event.
OFF disables new updates; it does not erase previously accumulated state.

Deletion uses the Reset-eligible parent without selecting on Reset outcomes.
The prefix is sequence[:boundary-1]. An ALL-containing prefix creates one
leave-one-layer-out branch per configured TTT layer, averaged within sample.
Otherwise the target is the layer of the first non-OFF prefix action. An unwritten
prefix is ineligible. Sham uses the lowest configured layer absent from the prefix
written-layer set, only when a real target exists. Both operations reset the target
local state to checkpoint; all downstream updates are recomputed. Sham must have
zero pre-state delta and native prediction/score parity. Full Reset results are
used downstream for the explained-fraction denominator, never for cohort selection.

K/P uses W_N = effective native boundary fast weights for the complete configured
TTT-state set, including base values for unused layers. Control uses an OFF prefix
and the identical frozen terminal suffix. K_N/K_C are native/control prefix keys
AND values for every attention layer. Native donor K/V are cloned and overwrite
positions [0, (boundary-1)*chunk_size) in the replay cache before attention. Neither
future nor generation K/V are transplanted. The entire prompt is replayed.

P0 injects all configured native effective fast weights, materializing all TTT
states. P1 injects only the union of layers used anywhere in the full native
sequence. Other layers take the original non-TTT path by temporarily removing
TTT attributes and restoring them afterward. P0 is not natural Control occupancy.
The cells are F00=(W_N,K_C,P0), F10=(W_N,K_N,P0), F01=(W_N,K_C,P1),
F11=(W_N,K_N,P1). Prompt structural traces and terminal reward are separate replays,
matching the formal smoke correction. No RNG reset is inserted into these operations.

Closure L1 covers recorded first-continuation and first-post-boundary-update
comparisons; L2 covers recorded final hidden states of full continuation chunks;
L3 covers recorded incomplete-tail comparisons; L4 is exact generation-start logits;
L5 is decoded prediction SHA256 equality; L6 is their conjunction. Empty groups
use Python all([])=True, as in the formal source. This proves only the recorded
closure criterion. It does not prove equality of every possible hidden state.
Historical raw tensors were not retained; result replay checks retained comparison
records, hashes, isolation fields and aggregation, not new raw-tensor comparisons.

R2 exhaustively evaluates A^tau B^(T-tau), A != B, tau=1..T-1. Constants and empty
regimes are excluded. Candidate count is A_count*(A_count-1)*(T-1). Every candidate
uses a clean full replay; no prefix reuse, cross-candidate state or stored deltas.
Select maximum terminal reward, then among scores within absolute 1e-12 choose
earliest tau, action-order A, action-order B, canonical schedule hash. This is a
sample-level hindsight optimum in the restricted family, not a controller or a
global optimum over all trajectories. With eight actions and T=8 there are 392.

Each configured statistical procedure retains its original implementation and RNG
ordering. Reverse and Reset distinguish mean per-sample retained gain from ratio
of summed gains relative to Sample Best. Deletion's formal aggregate explained
fraction is the ratio of sums over samples with positive Full Reset denominator;
its mean of per-sample ratios is a different statistic. Values are not clamped.
K/P primary effects use relative-logit L2 distances; reward summaries are secondary.
R2 formal retention is SUM(R2-OFF)/SUM(Dynamic-OFF), with paired and task-stratified
bootstrap. R2 uses population standard deviation for its gap summary whereas
Reverse/Reset use sample standard deviation. Frozen source and replay retain this.

Transplant V1 is not registered or exposed. The formal K/P prelaunch audit declares
WITHDRAWN_CAUSALLY_CONFOUNDED. Criticality and Observability are excluded from core V1.
