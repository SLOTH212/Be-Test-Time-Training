# Magnitude-matched control: execution and interpretation notes

This directory is an independent experiment. No historical evaluator or historical result was edited. `historical_source_hashes.json` freezes historical authorities, including winning trajectory source records; `experiment_source_hashes.json` freezes the new code/config before smoke. `IMPLEMENTATION_PLAN.md` records the pre-run audit and protocol. Do not edit frozen files to resume a run.

## Source roles

- `run_control.py`: frozen manifests, transparent native-prefix KV observer, existing evaluator integration, exact branch checks, smoke and formal worker.
- `calibration.py`: accepts only native/checkpoint tensors and sample/layer/seed identity; no scorer, reference, reward or trajectory input. Local Gaussian RNG never resets global RNG. Up to 57 trial norms; 0.1% search target, 1% acceptance. Actual BF16 displacement, not proposed FP32 displacement, determines matching.
- `pipeline.py`: smoke gate -> two workers -> source-integrity audit -> statistics. Any failed worker stops its sibling. It never starts Dynamic search.
- `summarize_new.py`: independent CPU aggregation and raw-record revalidation; A1 imported verbatim. Original historical summarize.py is never executed.

The Gaussian proposed direction is isotropic in FP32. BF16 rounding makes its represented displacement potentially anisotropic; this is not a claim that represented random offsets are exactly isotropic. Direction and scalar construction are reward-blind.

The runner uses native full-prompt replay with one layer's existing boundary-inject hook. No prefix donor transplantation is needed: the naturally reconstructed prefix K/V is hash-checked against native at all 28 attention layers in each branch. All five boundary before weights must equal stored Native and the four nontarget after weights must remain unchanged. Target after hash must equal calibrated W_random. Later weights/activations are recomputed, not constrained to match Native.

CPU affinity: worker0 8–15; worker1 16–23; orchestration/statistics 8–23. The historical tokenizer emits a regex advisory; tokenizer settings are deliberately identical to the frozen formal experiment, and Native tokenization/prediction/score parity is tested rather than applying an unrelated tokenizer change.

## Result files

`results/*.json` preserves full per-sample records, raw seed/branch intervention and KV hashes, predicted-output hashes, norm diagnostics, and sample aggregates. `raw_results.jsonl` is an aggregate convenience export; full execution evidence remains in `results/`. No generation outputs are fabricated from hashes.

`smoke/*.json` is validation only and excluded from formal statistics. Five structurally selected examples, all their 13 layer branches and five seeds, plus one repeated seed for each branch and historical-deletion replay. `aggregation_smoke_check/` validates the CPU aggregator schema using those five records; its BLOCKED status means an intentionally partial test fixture, not the formal experiment's status. Use root `final_report.md` / `pipeline_status.json` for formal status.

Scores in raw/example tables are fractions; report means/task-table score columns are percentages; contrasts use percentage points. The independent unit is example, never seed×branch. Bootstrap uncertainty is over examples conditional on the fixed five direction seeds, not an estimate over all possible random seeds. An inconclusive A3 CI does not demonstrate statistical equivalence.

## Existing-authority values

Native 93.24840764331209%; selective deletion 54.30785562632695%; A1 drop38.94055201698513pp, CI[35.167728237791934,42.579883227176204]. These numbers are read-only references. Magnitude matching is calibrated against each actual native/checkpoint weight pair, not against task scores or these summary numbers.

## Resumption

Use the original environment and verify source identity. `pipeline.py` reruns the gate and skips already completed atomic sample records only under the identical configuration. It has no permission to alter seeds/cohort/winners. If a failure exists, diagnose it before resuming; never remove failed cohort examples. Zero displacement, nonreproducible calibration, altered prefix/boundary or failed Native replay must remain explicit failures.

## Amendment 01: quantized closest fallback

The first pipeline stopped after72 completed samples at a 1.173097% residual norm error. This hard stop was stricter than the requested fallback. Original code/reports and failure evidence are preserved; attempt_01_preserved contains the partial exports. PROTOCOL_AMENDMENT_01.json freezes the72 retained result hashes. New *_closest.py entry points implement the already-authorized closest-candidate fallback with no numerical algorithm, seed, cohort, trajectory or score changes. Current status is pipeline_closest_status.json. Existing smoke evidence is reused because numerical construction/execution are unchanged. Strict <=1% attainment is reported separately from protocol-compliant closest matching; no assertion that every branch meets1% is permitted if exceptions exist.
