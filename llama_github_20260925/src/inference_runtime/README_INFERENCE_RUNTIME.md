# TTT Inference Runtime V1

This release packages the formal inference code lineage needed for Qwen3 Fixed_(N+2), full-action Sample Best, Dynamic Recomputed Lower Bound, aggregation, paired bootstrap, and integrity audits. It intentionally contains no model weights, benchmark JSONLs, training data, historical result directories, caches, logs, or environments.

Required composition:

```text
inference runtime + model authority + frozen RULER benchmark authority
```

A future HIT layout may be:

```text
/home/USER/data/zonghan/
  code/
  models/
  benchmarks/
  runs/
```

Replace `MODEL_PATH`, `BENCHMARK_ROOT`, and `RUN_ROOT` in a copied config template. The runtime validates model config/tokenizer presence, requested layer indices, benchmark manifest/index, 13-task/6500-sample inventory, context length, and available hashes. `CUDA_VISIBLE_DEVICES` is inherited; the package never selects every physical GPU automatically.

Workflow:

```text
fixed/run_fixed.py
  -> fixed/aggregate_fixed.py (Best Single + Sample Best)
  -> dynamic/run_dynamic.py (sample-exclusive workers, non-ceiling only)
  -> dynamic/aggregate_dynamic.py
  -> audits/integrity_audit.py
```

Planning is GPU-free:

```bash
python fixed/run_fixed.py --config configs/qwen3_1p7b_ruler16k_formal_reference.yaml --dry-run
python dynamic/run_dynamic.py --config configs/qwen3_4b_ruler32k_scaleup_template.yaml --dry-run
python run_pipeline.py --config configs/qwen3_8b_ruler32k_scaleup_template.yaml --dry-run
python tests/test_static.py
```

Historical 1.7B/16K is labeled `HISTORICAL_FORMAL_PARITY`. The 4B/8B 32K templates are separately labeled `SCALEUP_CONFIGURATION`; they use six configured single layers and therefore derive eight fixed actions including L30. Models and benchmarks remain external frozen authorities and must be recomputed per model: Fixed actions → Sample Best → non-ceiling cohort → Dynamic LB.

See `docs/SCIENTIFIC_SEMANTICS.md`, `docs/RESULT_SCHEMAS.md`, and `provenance/INFERENCE_SOURCE_AUTHORITY.json` before execution.
