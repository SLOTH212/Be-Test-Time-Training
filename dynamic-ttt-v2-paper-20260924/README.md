# Path-unified V2 TTT: final paper experiment evidence

This repository packages the **final old-checkpoint, 1K-chunk, path-unified V2** experiments and their completed CPU analyses. It includes Fixed7, Dynamic, Reverse, Reset, nontrivial Reset, Deletion, Sham, K-only continuation, exhaustive R2, and the completed equal-budget random baseline. **P factorial, V1 scientific results, the newly trained 2K model and cross-chunk pilot experiments are not primary experimental inputs and are not packaged.** Original V2 Stage-1/Stage-2 validation reports are included as validation history, not as the final Dynamic result.

## Results

Scores are on [0,100]; gains and intervals are percentage points.

| Scope | N | Sample Best | Sequential / Dynamic | Gain | 95% CI |
|---|---:|---:|---:|---:|---|
| Non-CWE 12-task parent | 6000 | 84.47139 | 85.97639 | 1.50500 | [1.24944, 1.77501] |
| Searched non-ceiling | 1433 | 34.98139 | 41.28286 | 6.30147 | [5.27448, 7.37849] |
| Full13 Case B composite | 6500 | 78.37667 | 79.76590 | 1.38923 | [1.15487, 1.63693] |

CWE Sample Best is 5.24. **CWE Dynamic was not measured:** its composite score is Sample Best by fallback, with zero incremental credit assigned by protocol. The 4567 non-CWE static-ceiling examples also carry their static score. The full13 composite is not 6500 measured Dynamic searches.

Dynamic improves 170/1433 searched examples; 1263 tie and none worsen. Mechanism cohorts are Reverse/R2=170, Reset/K=165, nontrivial Reset/Deletion=157, Sham=109. R2 exhausts 97356 candidates and matches 154/170 scores, including 62/78 native multi-switch cases.

On **only those 170 Dynamic-improved examples**, one frozen random realization with exactly 62772 nonconstant candidates scores 89.96078 versus Dynamic 93.52941 and Sample Best 40.41176. It recovers 93.28165% of Dynamic gain; Dynamic−Random=3.56863 pp, paired task-stratified CI [1.45098, 6.03922]. Random ties Dynamic on 159 samples and falls below it on 11. This is outcome-conditioned, answer-scored best-of-budget search; it does not estimate expected performance across random seeds or the full searched population.

## Recompute on CPU

Python 3.11 and NumPy 2.4.4 were used for the package verification. No weights, prompts, PyTorch or GPU are required for this command.

```bash
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python analysis/verify_package.py
python analysis/recompute.py --out outputs/recompute
python analysis/r2_supplement.py --out outputs/r2_supplement
```

The portable script independently rebuilds Sample Best from 45500 raw fixed cells, joins 1433 selected Dynamic winners, recomputes parent/searched/full13 paired bootstrap intervals, reconstructs eight mechanism contrasts from raw per-sample records, checks the entire R2 family coverage, and reconstructs Random best scores from all 62772 candidate records with exact frozen-plan matching. It compares results against frozen references and exits nonzero on mismatch. `outputs/` is ignored by Git. This is statistical reproduction from completed evidence, **not model inference replay**.

## Layout

- `analysis/recompute.py`, `analysis/verify_package.py`: portable CPU entrypoints.
- `data/fixed/`: complete 6500×7 raw fixed cells and convenience Sample Best table.
- `data/dynamic/`: complete searched winners, partition IDs and parent sample ordering.
- `data/mechanism/`: losslessly serialized full core and R2 per-sample JSON records, gzip-compressed JSONL.
- `data/random/`: full frozen trajectory plan, complete candidate results, budget/cohort/sample tables.
- `results/main.json`, `results/writer/`, `results/mechanism/`, `results/random/`: frozen tables, intervals and validation boundaries.
- `experiments/`: archived actual GPU execution scripts, protocols and copied runtime dependencies. These retain original source hashes/paths; **they are not a portable command-line runner**.
- `validation/`: original V2 exact prechecks and Stage-1/Stage-2 isolation evidence.
- `analysis/archival/`: original CPU structure audits and statistical scripts; some resolve original host paths or write beside their source. Do not run them as portable entrypoints.
- `provenance/`: original-to-package file hashes and external artifact inventory.

The additional R2 CPU entrypoint reproduces seven supplementary tables, including set-wise reversal/alignment intervals, task directions and task-centered correlations. See the [R2 CPU file index](docs/R2_CPU_ANALYSIS.md).

See [experiment coverage](docs/EXPERIMENTS.md), [external data and replay](docs/EXTERNAL_DATA.md), [validation boundaries](docs/VALIDATION.md), [Git upload instructions](docs/GITHUB_UPLOAD.md), and [source/licensing notes](THIRD_PARTY_NOTICES.md).

## Provenance and limitations

Trained checkpoint SHA256: `ca51ee90d20ee09efbe0b31dd21bf9faa70275508316d1ba7cb1387ee4f0725f`.
Final Dynamic executor SHA256: `85a659a6ab2d5c97187cd464db08e8074722fb7d4567284f04cb17671fd5f812`.
Original validated V2 executor SHA256: `8b7d8612e889c2a4c868ffa899acac1492c2ef75ab66fc39608e055c0371689c`.

The full-chunk/tail/generation functional forward is AST-identical in the original runner and final Fixed/Dynamic adapters. Original exact prechecks and final score/prediction bridges are separate evidence. A V2 empirical scorer-isolation intervention and a complete rerun of original prechecks under the final adapter hash were not found. Do not turn static source checks into numerical PASS claims. K closure only covers explicitly recorded sites, not all historical L1–L6 tiers.

R2 supports many score-matched low-switch alternatives; it does not establish universal shallow-to-deep dynamics, early formation→late exploitation, revisitation as the cause of failure, or broad timing tolerance. Search and mechanism claims remain conditional on reference scoring, this checkpoint, action family and chunk grid.

The package contains no model weights, dataset prompt texts, environment directories, manuscript PDFs or credentials. Archived source/report strings may retain original filesystem paths for provenance. No remote repository has been created or modified by preparing this package.
