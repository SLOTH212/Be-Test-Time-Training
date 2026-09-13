# Dynamic Test-Time Training

## Overview
An anonymous source release for contextual NTP test-time updates, fixed-action evaluation, Dynamic Recomputed Lower Bound (Dynamic LB), and causal mechanism experiments. Historical result tables describe Qwen3-1.7B. Qwen3-4B and Llama-3.1-8B implementations are provided with their separate scale-up and engineering scope.

**Release limitation:** CPU reproduction and sanitized result analysis work without private inputs. Exact training-data reconstruction still requires frozen historical anchors that are not distributed here. Trained checkpoints and benchmark inputs must be supplied separately. See [reproducibility](docs/REPRODUCIBILITY.md). Project-level licensing is pending; no repository-wide license is granted by this snapshot.

## Key Idea
The MLP down projection has a per-sample fast-weight state. A contextual next-token hidden target constructs an outer-product update. Each complete prompt chunk applies the current state before updating it. Dynamic LB searches chunk action sequences and recomputes their downstream execution.

## Repository Structure
`src/dynamic_ttt/` contains models, NTP, training, data reconstruction, evaluation, search, mechanisms, and statistics. `configs/` holds protocol settings; `tests/` uses CPU fixtures; `results/` contains sanitized scores and trajectories; `metadata/` records public artifact requirements; `docs/` explains protocols and reproduction.

## Installation
Use Python 3.11 or newer. The CPU suite was tested with Python 3.12, CPU PyTorch 2.8.0, and Transformers 4.57.3. Install an appropriate PyTorch build first, then:

```sh
python -m pip install -e '.[models,test]'
python -m dynamic_ttt.cli quickstart
python -m pytest -q
```

The quickstart enumerates actions, Sample Best, exact reverse, and the R2 family; it downloads no model or data. GPU training additionally needs the training extra and a compatible CUDA/Triton environment; it was not run on this CPU build machine.

## Data Preparation
Read [DATA.md](docs/DATA.md) before running `dynamic-ttt prepare-data`. The preserved builders are replay procedures with hash-checked prerequisites, not a claim that missing historical selectors were recovered. Raw text is excluded.

## Stage1 Training
Use the family-specific Stage1 YAML and the `dynamic_ttt.training.run` module under `torchrun`. It retains FSDP2, BF16, checkpointing, optimizer-window accounting, and DCP behavior. See [TRAINING.md](docs/TRAINING.md).

## Stage2 Training
Stage2 uses shifted labels and separate global optimizer-window answer/context denominators with context weight 0.1. Configure the Stage1 checkpoint and frozen Stage2 package before launch. See [TRAINING.md](docs/TRAINING.md).

## TTT Inference
Set `MODEL_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT` and select an inference JSON. Run `dynamic-ttt fixed --config configs/inference/qwen3_1p7b.json`. This requires local model and benchmark artifacts. [INFERENCE.md](docs/INFERENCE.md) specifies their layout.

## Fixed / Best Single / Sample Best
Fixed executes every configured action. Best Single selects one single-layer action globally. Sample Best maximizes over OFF, every single-layer action, and ALL for each sample. `dynamic-ttt sample-best --config CONFIG --input fixed.jsonl --output sample_best.json` aggregates complete fixed records.

## Dynamic LB
After Fixed, run `dynamic-ttt dynamic --config configs/inference/qwen3_1p7b.json`. The formal search uses beam 4, one forward sweep, full downstream replay, and no stored-delta stitching. [DYNAMIC.md](docs/DYNAMIC.md) describes ceilings, cohorts, replay accounting, and resume checks.

## Mechanism Experiments
`dynamic-ttt mechanism` runs historical Qwen Reverse, Reset, accumulated-component Deletion, K/P, or exact R2. Llama engineering omission and Reset V2 are explicitly separate. See [MECHANISMS.md](docs/MECHANISMS.md).

## Evaluation
The preserved per-sample scorer uses the frozen RULER matching procedure. Fixed covers 13 tasks / 6,500 samples. Historical Dynamic covers 12 tasks / 6,000 samples and excludes CWE. The reason for that exclusion has not been established as independent of outcomes; do not reinterpret the cohort as all 13 tasks.

## Statistical Analysis
`dynamic-ttt analyze --input results/per_sample/reverse.csv` produces descriptive means. The `dynamic_ttt.stats` modules also expose source-preserved paired and task-stratified bootstrap procedures; `analysis/mechanism_bootstrap.py` applies them to full mechanism result JSONL with explicit seeds. Descriptive means are not replacements for the frozen inferential procedures.

## Reproducing Results
See the [result-to-command map](docs/REPRODUCIBILITY.md). Public tables are sanitized historical evidence, not outputs of new GPU runs performed for this release. Never mix 6,500-sample Fixed, 6,000-sample Dynamic, 1,435 searched, and 173 improved cohorts.

## Checkpoints
No weights or DCP state are included. Stage1 FINAL and Stage2 FINAL distribution locations remain unassigned. See [CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Tests
Run `python -m pytest -q`. Tests use tiny models and CPU fixtures for update order, tails, generation, reset/isolation, actions including L30, mechanisms, scoring, bootstrap, and Stage2 masks. Full distributed training and formal inference are `NOT_RUN_LOCALLY_REQUIRES_GPU`.

## Known Protocol Notes
Historical Deletion removes an accumulated component; Llama engineering omission skips one update. Reset V2 is not paper-frozen. Exact one-switch R2 excludes constants and endpoints, supported by the later formal R2 authority. [PROTOCOL.md](docs/PROTOCOL.md) and [MECHANISMS.md](docs/MECHANISMS.md) define these distinctions.
