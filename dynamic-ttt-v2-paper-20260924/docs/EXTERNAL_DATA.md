# External artifacts and GPU replay

The Git repository is sufficient for packaged-result CPU recomputation. It is not sufficient for inference replay by itself.

| Outside Git | Why / how to use |
|---|---|
| Trained 1.7B checkpoint and tokenizer/config | Exact trained checkpoint, not untrained Qwen3 base. Verify model SHA256 in README and original protocols. Publish separately in a model/data repository if desired. |
| Full benchmark sample manifest / prompts / references | Required for inference and reference scoring. Exact manifest SHA256 is in the external inventory and Fixed protocol. Distribute separately with upstream dataset terms and provenance. |
| All 1433 full Dynamic search records | Selected winners are in Git. Full 531113-candidate landscapes remain external; exact path/size/SHA256 are listed in `provenance/EXTERNAL_ARTIFACTS.json`. Required to independently re-audit every R2 reused candidate against Dynamic search. |
| Original Fixed individual execution JSONs | Full fixed cell scores/hashes are in Git; original detailed per-run execution records stay in the completed source run. |
| Validation full tensor-comparison traces / process ledgers | Compact validation CSVs/reports are in Git; detailed original trace records remain external. |
| Random execution START/END ledgers | Frozen plan and all candidate results are in Git; original ledgers and full-run postflight report provide exactly-once process evidence. Only the report is copied into Git. |

`provenance/EXTERNAL_ARTIFACTS.json` uses paths relative to the original `ttt` workspace as identifiers, not public download URLs. No download endpoint has been invented. GPU reproduction is **external-data and path-configuration dependent**, not advertised as one-command ready.

Archived runtimes are preserved byte-for-byte. Some imports and `Context` setup refer to the original host. Before any GPU rerun, configure model/data/output paths in a separate working copy, review required dependencies, and run the original parity/validation gates. Do not rewrite frozen evidence in this repository or mix modified-executor results into these tables.

`provenance/COPIED_FILES.json` maps byte-identical files and losslessly serialized/compressed sources. `data/dynamic/winners.jsonl` is a projection of the selected-winner fields from each full Dynamic record; omitted event traces remain in the hash-pinned original. `results/main.json` selects final V2 fields from the completed writer audit. Gzip JSONL streams preserve numerical values but are not byte-identical concatenations of original pretty-printed JSON files.
