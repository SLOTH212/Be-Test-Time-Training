# Experiment coverage

| Experiment | Included data | Actual execution source | Scope |
|---|---|---|---|
| Fixed7 | `data/fixed/FIXED7_FULL6500.csv` | `experiments/fixed/` | 6500 samples, 45500 cells; reused Stage-2 V2 cells plus completed missing cells |
| Dynamic | `data/dynamic/winners.jsonl` | `experiments/dynamic/` | 1433 searched, 6000 parent; constant retention, beam4, one forward sweep |
| Reverse / Reset / Deletion / Sham / K | `data/mechanism/core.jsonl.gz` | `experiments/mechanism/` | 170 / 165 / 157 / 109 / 165; P omitted |
| Exact R2 | `data/mechanism/r2.jsonl.gz` | `experiments/mechanism/` | 170, 97356 candidates; 7733 reused current V2 candidates |
| Random | `data/random/RANDOM_RESULTS.jsonl.gz` | `experiments/random/` | 170 improved only; 62772 candidates, one seed, equal per-sample nonconstant budget |
| CPU R2 structure | `analysis/archival/r2/` | same directory, original host scripts | Complete base and supplementary numeric tables; plot scripts archived, figures not required |
| Executor tests | `validation/stage1/`, `validation/stage2/` | original scripts + sources | 144 precheck trajectories, 1384 Stage-1 and 11480 Stage-2 trajectories |

No additional inference was performed for packaging. The stage validation reports retain historical replay comparison context; their 173/1435 result cohorts must not replace the final 170/1433 cohorts. The final release source still needs external inputs and path adaptation for GPU reruns. CPU reproducibility is provided by `analysis/recompute.py`.

## Bootstrap rules

Main Dynamic: seed20260921, PCG64, 20000 replicates, task lexicographic, sample order from `data/dynamic/fixed_all6500.jsonl`, batch1000, NumPy linear quantiles. Full13 retains the CWE zero-delta stratum and its RNG draws. Mechanism: same seed/draw count, task strata, original COHORT order. Random: seed20260923, 20000 replicates, task then sample-ID ordering, full index matrix per task, linear quantiles. CPU R2 structure uses its own recorded seed20260923 with 4000 mean / 2000 correlation draws.
