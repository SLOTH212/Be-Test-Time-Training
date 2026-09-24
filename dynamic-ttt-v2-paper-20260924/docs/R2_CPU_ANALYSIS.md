# R2 CPU analysis index

Both the base CPU audit and its completed numerical supplement are included. They use the final V2 170-improvement cohort and complete 97356-candidate R2 enumeration. No old V1 R2 is statistical input.

| Analysis | Included file, relative to `analysis/archival/r2/` |
|---|---|
| Main interpretation and limits | `CPU_AUDIT_REPORT.md` |
| Base summary / recovery / multiplicity / prediction match | `summary.json`, `supplementary_summary.json`, `per_sample_improved.csv` |
| All task summaries | `per_task.csv` |
| Full boundary landscape, success sets, plateau lengths | `pair_landscapes.csv`, `boundary_details.csv` |
| Count-matched forward/reverse pairs | `count_matched_reversals.csv` |
| Native trajectory complexity and gains | `gain_complexity_correlations.csv`, `gain_by_switch_count.csv` |
| Base action-pair / alignment statistics | `transition_enrichment.csv`, `modal_alignment.csv` |
| Candidate completeness and search ordering limits | `r2_candidate_coverage.csv`, `candidate_coverage_all_searched.csv`, `search_log_coverage.csv`, `search_order_provenance.json` |
| Supplementary findings and boundaries | `completion/COMPLETION_REPORT.md` |
| All best/matching sets; alignment and reversal CIs | `completion/setwise_alignment_and_reversal.csv`, `completion/setwise_summary.csv` |
| Task-specific transition direction and baseline | `completion/task_transition_enrichment.csv`, `completion/task_transition_types.csv` |
| Cross-task direction consistency | `completion/dominant_transition_by_task.csv`, `completion/cross_task_direction_consistency.csv` |
| Within-task centered rank correlations | `completion/within_task_rank_correlations.csv` |
| Task structure comparison | `completion/task_structure_comparison.csv` |

Writing-facing exact numbers and definitions are in `results/writer/TABLE_R2_STRUCTURE_V2.tsv` and `results/writer/R2_STATISTICAL_DEFINITIONS.json`. Full raw R2 candidate records are `data/mechanism/r2.jsonl.gz`; Native trajectories and cohort are `experiments/mechanism/COHORT.json`.

## Portable supplementary reproduction

```bash
python analysis/r2_supplement.py --out outputs/r2_supplement
```

This CPU-only adapter preserves the original supplementary numerical formulas while replacing host-specific paths and the input loader with repository-relative compressed R2 records. It reconstructs all best/matching sets and independently reproduces **seven tables**, including alignment, set-wise reversal and their confidence intervals, task transition enrichment, direction consistency and centered-rank correlations. It asserts equality with the frozen tables (numerical tolerance1e-10). Original pretty-printed input hashes remain in `provenance/COPIED_FILES.json`; temporary expanded JSON files have their own serialization hashes, not falsely labeled original hashes.

This entrypoint uses the packaged base per-sample structure CSV for task-centered correlations. It does not claim to re-audit all531113 original Dynamic candidate records, which remain external. The separate `analysis/recompute.py` checks complete R2 candidate coverage and reconstructs R2 aggregate scores/drop intervals from raw R2 records.

The original base audit, report, supplementary analysis and plotting source scripts are all retained under `analysis/archival/r2/`. Original scripts retain host paths and should not be run in-place. Publication figures are not included in this numeric evidence package; plotting scripts and prior plot status are preserved. The original complete curve atlas was not delivered and is not represented as completed.

## Interpretation boundaries

- 62/78 native multi-switch samples have a score-matched R2; 16 do not.
- 50/62 have multiple matches; only22/62 have adjacent successful boundaries within the same ordered action pair;40/62 are isolated within pair.
- At least one prediction-hash match exists for60/62 matched multi-switch samples; this is not state equality.
- Set-wise reversals remain selected-set comparisons, not unbiased causal effects.
- Early/late alignment does not establish early formation→late exploitation; direction is not uniformly shallow→deep across tasks.
- Revisitation is not established as the cause of R2 failure; gain is not established to require more switches.
