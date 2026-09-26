# Improved-only V2 mechanisms, 2026-09-26

User scope: Llama-3.1-8B (39 improved samples), Qwen3-4B (85 improved samples), no CWE. GPUs 0,1,2,3,4,6,7. No P factor; no Deletion. Existing services are not stopped.

Protocol: Native replay must exactly match committed Dynamic score and decoded-prediction hash. Reverse reverses the complete-chunk action sequence. Reset eligibility uses the existing resolution-aware V2 screen, frozen explicitly for this new run; this differs from the old zero-eligible 4K configuration. The reset boundary is the start of the terminal constant-action run, not necessarily the first switch. Reset replaces all six effective fast-weight states with checkpoint weights before that chunk; prefix context is retained.

KV: Capture native boundary effective weights and native prefix keys/values. Use an OFF-prefix control with the native suffix actions. Run and save an independent Control branch before K0/K1. Both KV cells inject identical native effective weights in all six TTT layers. K0 uses the recomputed control prefix KV; K1 replaces prefix KV at every attention layer with native KV. Require K1 prediction, final effective-state hashes and recorded suffix trace to equal native replay. Save per-layer donor/replacement key/value hash, shape, dtype and prefix length; capture the donor prefix to CPU after Native generate. Repeat Native after interventions on EVERY sample. P0/P1 is not varied. KV is evaluated on Reset-eligible trajectories only.

R2: All ordered distinct action pairs and every interior boundary, including all six layers and ALL/OFF. No constant schedule, no early stopping after score 1. Winner tie order: smallest boundary, canonical action A order, canonical action B order, sequence hash. Every replay recomputes the full prompt. No stored-delta stitching, tail updates or generation updates. Model weights remain unchanged.

Implementation extends a freshly compiled copy of the existing V2 MLP function with boundary capture/injection only. Existing sources, weights, Fixed8 and Dynamic receipts are unmodified. Native and three R2 schedules are compared to the unmodified V2 numerical loop in smoke; eight constants are compared with recorded Fixed8 hashes and scores. Repeat after interventions checks sample/branch isolation. KV prefix copies and Reset state assignments are checked exactly.

This selected cohort is conditioned on Dynamic improvement. Mechanism effects on it must not be presented as full-benchmark causal effects. R2 selects with reference-answer scoring and is an oracle search.

Formal completion: both model phases must have all expected per-sample FINAL receipts, all 7 workers exit 0, unchanged base hashes. Partial R2 records are progress, not final results. A failed worker halts its model phase and blocks subsequent phases.

User confirmed on 2026-09-26: retain resolution-aware 4K eligibility. The strict 1.7B first-switch/T <= 0.25 rule selects ZERO samples for these 6/7-chunk prompts; report this explicitly alongside adapted cohort counts 30/69. Reference handoff archive SHA256: 218889f5e22726923e70ff08d26abb55a3651cda77bc0657c7efd7b63b13282b.

Trace coverage: all layers, first full suffix chunk and remainder (if present), attention input and q/k/v projections and attention output, MLP input/output and residual; final norm entire suffix and actual returned prefill logits. Generation activation trace is not covered. Actual TTT update invocations are observed, with no generation calls and exactly 4095 valid pairs per full 4096-token chunk.
