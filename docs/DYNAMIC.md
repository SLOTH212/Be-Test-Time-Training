# Dynamic Recomputed Lower Bound

The actual search implementation is `dynamic_ttt.dynamic.search`; the configured wrapper binds action/layer/chunk settings. Beam width is four, one sweep, forward positions. All constant action sequences seed the search. A candidate is executed from fresh sample state, recomputing downstream updates. Cached candidate scores avoid redundant search evaluations; accumulated deltas from unrelated paths are never stitched.

The replay-budget helper bounds unique candidate evaluations; final trace capture is accounted separately. Scores, deterministic tie handling, and winning schedules come from the preserved source. The public runner writes committed per-sample JSON atomically with configuration identity, completed flag, prediction hash, and replay count. A per-sample FileLock prevents simultaneous writers and releases after process exit. Resume rejects incompatible committed records.

Historical ceiling samples are not searched. The 1,435 released searched trajectories exclude raw prompt/prediction text; their schema is `schemas/dynamic_trajectory.schema.json`. The runner's operational result JSON contains additional fields and is not itself the archival trajectory schema. Convert explicitly before archival validation. The formal Dynamic cohort excludes CWE; do not silently add it or reuse the 6,000-sample bootstrap for a different benchmark.
