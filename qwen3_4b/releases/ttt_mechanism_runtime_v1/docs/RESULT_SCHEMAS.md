# Result schemas

Portable commits use MECHANISM_SAMPLE_V1: model_identity, benchmark_identity,
sample_id, task, benchmark_global_index, parent_dynamic_trajectory,
parent_dynamic_score, sample_best, intervention, intervention_score,
config_authority, errors, timing, status=COMMITTED, payload_sha256.
Experiment-specific fields live under details.reverse/reset/deletion/kp/r2.

Reverse retains the transformed sequence and evaluator record. Reset retains exact
screening, boundary, per-layer reset events and evaluator record. Deletion retains
selected branches, their within-sample mean, optional matched sham, and the matched
Full Reset score needed for downstream explained fractions. K/P retains all four
cell records, trace comparisons, native reward, K/P assignment and closure levels.
Its intervention_score is F11 reward for schema uniformity only; primary K/P
analysis uses all four relative-logit distances. R2 retains every candidate and the
winner, ties, OFF score and restricted-family count within the atomic sample.

Historical schemas remain preserved in original source/configs and compact fixture
records. They are not overwritten or relabelled as new portable commits. Historical
replay explicitly interprets each original schema, including distinct comparison
field names and commit statuses.
