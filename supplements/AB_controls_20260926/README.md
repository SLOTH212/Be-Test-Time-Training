# Additional formal V2 mechanism controls A and B

Copy this `supplements/AB_controls_20260926` directory into the existing Be-Test-Time-Training repository and commit the directory contents. This is an additive supplement, not a replacement for the main release. Do not commit the outer transfer archive or restored analysis workspace.

## Frozen experiments

- **A — Magnitude-matched nonzero perturbation:** 157 nontrivial-reset examples, 349 layer branches, 5 seeds (1,745 random interventions). Native 93.2484%, deletion 54.3079%, random 58.8497%. Native minus random 34.3987 pp [31.5006, 37.2451]; random minus deletion 4.5418 pp [1.6709, 7.3474]. Matching <=1%: 1,744/1,745; one closest-candidate exception 1.1731%. Current entrypoints are `pipeline_closest.py`, `run_control_closest.py`, `summarize_closest.py`. Original attempt and amendment are preserved; the original failed pipeline status does not describe the completed amended run. Read `pipeline_closest_status.json`, `PROTOCOL_AMENDMENT_01.json`, and final reports.
- **B — Within-chunk target-alignment control:** 170 Dynamic-improved examples, 5 seeds. Native 93.5294%, Sample Best 40.4118%, B1 shuffle 46.9353%, B2 shuffle with attempted magnitude matching 46.9843%. Native minus B1 46.5941 pp [43.7705,49.3157]; Native minus B2 46.5451 pp [43.8196,49.1373]. Current entrypoint `pipeline.py`. B2 only 8,592/18,800 writes meet <=1% norm error; maximum error 71.1001%. Execution/protocol PASS is **not** strict magnitude-match PASS. The equal-magnitude causal alignment claim is NOT ESTABLISHED. B1 measures the total effect of target-pair shuffling, including changes in update magnitude.

A and B have different cohorts and interventions; do not pool them. Bootstrap unit is example, with branch/seed aggregation defined in each frozen script; 20,000 task-stratified paired draws. No new GPU experiment was run to create this package.

## Contents and provenance

`payload/runs/formal/` preserves original experiment source, configuration, frozen cohorts/trajectories/seeds, complete per-example raw results, smoke records, final tables, write-level diagnostics, source hashes, logs and failure/amendment evidence. Historical input files named in both frozen historical-source manifests are included at their original relative paths under `payload/`. Large JSON/JSONL/log files are individually gzip-compressed without changing their restored bytes. Readable reports and CSV tables remain directly accessible. `FILE_MANIFEST.json` maps each file to its original absolute path and SHA256. `SHA256SUMS` covers packaged files.

Excluded: weights, installed environments, pyarrow binaries, Python bytecode, and redundant aggregation test fixtures. Full B per-write diagnostics and candidate metadata ARE included. Historical benchmark samples are replication inputs; retain the original dataset/model licenses and main repository notices. This supplement grants no new third-party license.

## CPU reproduction (no model/GPU needed)

Use Python 3.11 with the pinned analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python recompute.py --verify-only --output /tmp/unused-ab-check
python recompute.py --output /tmp/ab-controls-recomputed
```

Choose a NEW output directory (about 1 GB free). The helper verifies/decompresses original bytes and runs the original CPU summarizers on restored copies. It redirects only historical input reads to the restored tree, without editing frozen config or source files. It requires exact equality of both recomputed bootstrap-summary objects to the archived summaries; the environment includes NumPy version in B's summary, so use the pinned version. It writes `CPU_RECOMPUTE_CHECK.json` and regenerated tables in the output directory, never the evidence payload.

## GPU replay limitations

The original GPU runners retain historical absolute paths, checkpoint/data dependencies and CPU affinity 8–23 (workers 8–15 and 16–23). This archive is a source/evidence supplement plus a portable CPU analysis entrypoint, not a turnkey GPU environment. GPU reproduction requires the checkpoint whose hash is frozen in config, the original V2 runtime/environment and restoration or explicit porting of historical paths. Ported execution must be a new run with a new provenance record; do not edit this frozen evidence or claim the old hashes validate edited source. See each experiment's REPRODUCTION_NOTES.md and environment records.
