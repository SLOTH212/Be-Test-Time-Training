# TTT Mechanism Runtime V1

This package recovers the completed Qwen3-1.7B formal mechanism chain and provides
configurable Qwen3 execution adapters. It depends on the separately frozen
`ttt_inference_runtime_v1` archive, SHA256
`65a0d7bd9c8801c92fc30477ab8662d3eb52261612eff2acef0bbeb3edf6ba51`.
No inference, training or remote transfer is performed by packaging or tests.

Workflow: frozen inference runtime + model + benchmark -> complete Fixed results
-> Sample Best -> completed Dynamic results -> deterministically derived improved
cohort -> mechanism-specific screening -> mechanism execution -> aggregation.
Mechanisms are downstream of Dynamic. They do not launch another Dynamic search.

Verified historical cohorts: Reverse 173; Reset 167 eligible of 173; K/P the same
167; Deletion 151 of the Reset-eligible 167; R2 173. Deletion and K/P are sibling
children of Reset screening. Reverse outcomes do not select Reset samples.

Set configurable dependency paths before any command:

```bash
export TTT_INFERENCE_RUNTIME_ROOT=/your/path/ttt_inference_runtime_v1
export TTT_INFERENCE_RUNTIME_ARCHIVE=/your/path/ttt_inference_runtime_v1.tar.gz
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES='' python tests/level0.py
CUDA_VISIBLE_DEVICES='' python tests/level1.py
python audits/verify_package.py
python run_mechanism.py --config configs/qwen3_1p7b_reverse_formal_reference.yaml
```

Tests use the Python standard library, NumPy, and CPU PyTorch. No package install
is performed. Model execution additionally needs the frozen inference runtime's
Transformers/PyTorch environment. Level 0 tests configuration, screening boundaries,
cohort reconstruction, R2 enumeration, dependency rejection and process ownership.
Level 1 tests tiny local tensor/cache operations and replays compact historical
statistics including the original 20,000-resample bootstraps. It loads no model
checkpoint and invokes no model generation. Audit status files describe actual
executed checks rather than future inference validation.

Historical reference configs preserve source configs as immutable evidence while
replacing machine paths with placeholders in the runnable interface. Replace all
input/output identity and path placeholders for a new run. `fixed_results` is a
JSONL with sample_id, mode/action, score. `dynamic_results` is either original formal
results directory or a JSONL with sample_id, task, dynamic_score, sample_best,
action_sequence, chunk_count, search_status=SUCCESS and prediction_hash. Inputs
must be complete, unique and consistent. Benchmark sample_manifest.jsonl provides
canonical ordering, prompts, references, task families and generation budgets.

The 4B config is `SCALEUP_TEMPLATE_NOT_FORMAL_FROZEN`. Select one mechanism, its
experiment-specific statistical procedure, paths and real authority hashes before
execution. The template includes layers [0,6,12,18,24,30], eight actions, context
32768 and chunk4096; it does not freeze future results or cohort sizes.

Future execution requires an explicit flag:

```bash
python run_mechanism.py --config /path/to/frozen_run_config.yaml --execute --workers 2 --worker-id 0
python run_mechanism.py --config /path/to/frozen_run_config.yaml --execute --workers 2 --worker-id 1
python aggregate_mechanism.py --run-root /path/to/new_run
```

Do not target historical run directories. Each worker owns disjoint ordered
samples. A permanent per-sample kernel lock prevents concurrent execution; locks
release after process exit. Full sample results commit atomically with payload and
configuration hashes. Resume rejects changed config, cohort, runtime source or
invalid commits. Incomplete samples replay from clean state; committed samples are
not repeated. This generic orchestration does not use FSDP. Deletion additionally
requires matched Reset commits through `reset_results`; these outcomes are joined
only after structural selection.

`provenance/original/` preserves exact original source bytes and historical paths;
never launch those files directly on a new host. `runtime.py` and `execution.py`
select verified definitions and parameterize execution settings without modifying
sources. The full historical master replay entry point is
`tests/replay_historical.py`; it requires the original formal trees. Portable
Level 0/1 instead use only bundled compact fixtures and the explicit dependency.

See [semantics](docs/SEMANTICS.md), [result schemas](docs/RESULT_SCHEMAS.md), and
[HIT deployment](docs/HIT_DEPLOYMENT.md). Llama mechanism support is deferred.
