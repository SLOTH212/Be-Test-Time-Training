# File changes

No historical source, cohort, winner manifest, checkpoint or formal result was modified. No shared evaluator hook was changed. Reused historical run.eval_branch and executor.load_model/Evaluator, existing target-only inject hook and original scorer.

All new files live in this directory. Main added code: calibration.py; run_control.py; pipeline.py; summarize_new.py; amendment entrypoints run_control_closest.py, pipeline_closest.py, summarize_closest.py. Planning/provenance: IMPLEMENTATION_PLAN.md, config.json, cohort_manifest.jsonl, branch_manifest.jsonl, historical_source_hashes.json, experiment_source_hashes.json, PROTOCOL_AMENDMENT_01.json, amendment_source_hashes.json, REPRODUCTION_NOTES.md.

Required final deliverables: config.json, cohort_manifest.jsonl, branch_manifest.jsonl, raw_results.jsonl, example_level_results.csv, task_level_results.csv, bootstrap_summary.json, magnitude_match_diagnostics.csv, audit_report.md, final_report.md. Full raw execution evidence: results/*.json. Validation evidence: smoke/*.json, SMOKE_RESULT.json, CPU_CALIBRATION_SELF_TEST.json, source_integrity_check.json, environment_*.json, WORKER_*_DONE.json, quantization_exceptions/*.json. Preserved partial attempt and failure artifacts remain for audit.
