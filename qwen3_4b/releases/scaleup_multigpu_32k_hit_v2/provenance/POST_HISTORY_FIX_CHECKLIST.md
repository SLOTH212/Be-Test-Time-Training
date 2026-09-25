# Post-History Fix Checklist

- [x] Exact record cursor and cumulative-token resume without skip/duplicate — canonical cursor/rank state in `distributed_train_worker.py`; partition/resume Level 1.
- [x] Complete Python, NumPy, torch CPU, and per-rank CUDA RNG — `distributed_checkpoint.py`; schema and exact-next-value Level 1.
- [x] Atomic A/B checkpoint rotation, hashes, and validation before authority update — partial directory, fsync, rename, file hashes, hashed `latest.json`; Level 1 reload.
- [x] Full optimizer/scheduler restore with device-correct optimizer state — DCP optimizer plus per-rank scheduler; CPU Level 1 and CUDA Level 2.
- [x] Immutable, hashed Stage1 parent publication — collective full-state consolidation, refuse overwrite, per-file hashes, atomic promotion.
- [x] Stage2 model-only load and optimizer/scheduler/step/cursor/RNG reset — strict full-state load before fresh Stage 2 state; Level 1/2 handoff.
- [x] Stage2 answer/context objective with independent global denominators — `count_window` and two independently scaled fused numerators.
- [x] Token alignment and record/document boundary assertions — declared-token checks and explicit per-record boundaries in the distributed worker.
- [x] Validation performs no optimizer update and does not mutate parameters — no-grad validation and parameter-version assertion before Stage 1 publication.
- [x] Inference tail behavior remains separate and unmodified — legacy inference/evaluation files are byte-identical from base; distributed additions do not import them.
- [x] TTT gate/projection/state coverage and zero convolution-forward contract — frozen recovered training model plus Level 0 five-layer parameter inventory; legacy runtime tail tests retained.
- [x] Formal lifecycle/startup, failure, and exactly-once promotion safety — rank-zero shared ownership, all-rank barriers/failure primitive, refusal to overwrite parent, existing lifecycle scripts retained.

This checklist must be re-audited after implementation; checking a box requires source and test evidence.
