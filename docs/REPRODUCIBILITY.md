# Reproducibility map

## Available now
Installation, CPU tests, quickstart, schema validation, and descriptive analysis need no private model/data. Historical source/public parity fixtures compare tensor updates, actions and Sample Best, mechanism definitions, seeded bootstrap, and scoring. These checks do not establish end-to-end GPU parity.

| Result | Config | Command / procedure | Expected output |
|---|---|---|---|
| fixed_baseline | inference/qwen3_1p7b.json | dynamic-ttt fixed --config CONFIG | configured run root / fixed commits; reference results/aggregate/table_fixed7_6500.csv |
| sample_best | same | dynamic-ttt sample-best --config CONFIG --input fixed.jsonl --output sample_best.json | sample_best.json; complete actions required |
| dynamic_lb | same | dynamic-ttt dynamic --config CONFIG | dynamic/results/*.json; reference results/aggregate/table_main_1_sequential.csv |
| reverse | mechanisms/reverse.json | dynamic-ttt mechanism with cohort/benchmark/fixed/output arguments | per-sample JSON; reference results/per_sample/reverse.csv |
| reset | mechanisms/reset.json | same mechanism command with eligible cohort | per-sample JSON; reference results/per_sample/reset.csv |
| kp_factorial | mechanisms/kp.json | same with K/P cohort | per-sample JSON; reference results/aggregate/table_main_3_kp.csv |
| deletion | mechanisms/deletion.json | same with accumulated-component cohort | per-sample JSON; reference results/per_sample/deletion.csv |
| r2 | mechanisms/r2.json | same with improved cohort | per-sample JSON; reference results/aggregate/r2_main.csv |

All config paths in the table are relative to `configs/`. GPU commands need the artifacts and layouts documented in DATA, INFERENCE, MECHANISMS, and CHECKPOINTS. Output CSV conversion is an explicit analysis step; these commands do not claim to regenerate camera-ready tables automatically.

## Statistical procedures
The public CSVs can be inspected with `dynamic-ttt analyze --input FILE`. Full execution JSONL can be aggregated with `python analysis/mechanism_bootstrap.py --mechanism reverse --input results.jsonl --seeds configs/mechanisms/reverse_statistics.json --output summary.json`. Supply the frozen experiment's seeds and resample count; do not invent seeds when claiming exact confidence-interval parity. `stats.dynamic` retains the historical 12-task/6,000-sample procedure and must not be applied to differently sized cohorts.

## Outstanding reproduction requirements

For the complete 6,000-sample Dynamic cohort, run `python analysis/dynamic_results.py --input results/per_sample/sequential.csv --output dynamic_summary.json`. This validates the full twelve-task population and Sample Best before applying the preserved formal bootstrap, including ceiling zero-delta samples. Defaults retain seed 20260828 and 10,000 resamples. Scores are fractions; the `dynamic_minus_sample_best_pp` field is in percentage points.

1. Exact training-data replay needs historical anchors and ordering manifests that are not distributed and whose original selector/RNG is not recovered. See DATA.md.
2. Stage1/Stage2 checkpoint download locations and a lawful benchmark reconstruction/package need user publication.
3. Full GPU/distributed integration is NOT_RUN_LOCALLY_REQUIRES_GPU. Scale-up engineering support does not imply new formal result validation.
4. Project-level license remains NEEDS_USER_DECISION. Third-party notices apply only to their respective components.

These limitations prevent describing this snapshot as fully public-source reproducible. No scientific algorithm was substituted to hide missing inputs.

To reproduce inferential summaries directly from sanitized CSVs, run `python analysis/reverse_results.py --input results/per_sample/reverse.csv --seeds configs/mechanisms/reverse_statistics.json --output reverse_summary.json` (substitute reset, deletion, kp or r2). These scripts retain the original estimators and explicit denominator policies. They resample in released CSV row order; finite-bootstrap endpoints need not equal historical table endpoints if the original row order differed.
