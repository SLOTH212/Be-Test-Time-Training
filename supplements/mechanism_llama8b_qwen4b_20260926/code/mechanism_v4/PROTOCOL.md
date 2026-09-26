# V4: original KV restored, historical Selective added, eight-GPU recovery

User explicitly reverted the proposed four-boundary KV sweep before its formal
launch. V3 smoke was cancelled; its artifacts are engineering history only.
This V4 is an independent immutable successor of the original seven-GPU V2.

## Original mechanisms and KV

Native, Reverse, Reset, KV and R2 are numerically and scientifically identical
to V2: exact copied Engine, unchanged model/tokenizer/data/scoring/actions and
per-sample generation budgets. KV uses the last-switch boundary, identical
Native fast weights in K0/K1, Control versus Native prefix KV and no P factor.
The 4K resolution-aware eligible cohort remains Llama30 and Qwen69. This retains
the documented screening difference from the historical 1.7B quarter rule.
All completed original mechanisms including KV are verified and imported.
There are no four-boundary tasks and no new KV boundary selection.

## Selective: migrated 1.7B protocol adapted to six layers and 4K

Parent cohort is Dynamic-improved Llama39/Qwen85. Use original resolution-aware
Reset eligibility, then the migrated historical deletion_plan: last-switch
boundary; if prefix contains ALL, delete each of six layer components one at a
time and average within sample; otherwise delete the first non-OFF prefix
action's layer. No written prefix means ineligible. SHAM is the lowest layer
never written in the prefix when one exists. No filtering on Reset outcome.
Llama29 eligible:64 target branches and22 SHAM. Qwen62:117 targets and51 SHAM.
Candidate layers0,6,12,18,24,30 replace historical five-layer list.

Deletion restores only the selected effective down-projection fast weight to
checkpoint at the boundary. Preserve Native prefix KV and other layer states,
retain future actions/updates (state may reaccumulate). This removes accumulated
parameter state, not one update and not historical KV. Require true targets
nonzero, SHAM exactlyzero and exact Native closure. ALL-prefix LOO must leave
at least one other nonzero state component. Also run no-op reinjection and a
same-boundary FullReset; report selective mean/effect and explained fraction
when Native minus Reset is positive, without clipping.

Exact CPU reference parity covers124 actual samples and4648 trajectories.
GPU smoke covers ALL-prefix and SHAM for both models plus unmodifiedV2 parity.
Original KV numerical code is byte-identical to already-GPU-validated V2 and
its prior smoke provenance is retained, not relabeled as new smoke.
Recorded traces cover prefill firstsuffix/remainder, suffixfinalnorm and returned
prefill logits, not all generation activations. No generation/tail writes.
Full model hash after each task; per-replay parameter version checks.

## Safe recovery

Original seven-card workers continue during GPU5 smoke. After both models pass,
identity-check/freeze old manager, stop only its workers, confirm all reaped,
snapshot+hash original files, validate candidate enumeration and completed
FINAL/tiebreaks. Preserve all committed and every persisted8-candidate block.
Original result files remain untouched; write only a supersession receipt.
R2 uses32-candidate atomic tasks with eight-record checkpoints, exclusive flock
held for each task and work-sharing within model. Separate model processes.
No duplicate concurrent task or changed candidate/scoring/scientific policy.
Eight authorized GPUs0..7; retain all unrelated services including GPU5.
At least35000MiB free and successful actual smoke required before formal start.
