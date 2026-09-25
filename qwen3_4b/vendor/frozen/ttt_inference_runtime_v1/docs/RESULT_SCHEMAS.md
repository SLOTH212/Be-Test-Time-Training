# Result schemas

Fixed sample records contain `sample_id`, `task`, benchmark index, model and benchmark identities, configured action, numeric score, generated output, status/error, timing, and config identity. Result commits use temp-write, flush, fsync, atomic replace, and parent-directory fsync.

Fixed aggregation derives actions as `OFF + L<configured layer> + ALL`. Best Single is the globally best single-layer action. Sample Best is the per-sample maximum over every configured fixed action, including OFF and ALL.

Dynamic sample records contain `sample_id`, `task`, Sample Best, actual Dynamic score, action sequence, unique-candidate replay count, complete chunk count, ceiling/search status, errors, prediction hash, source hash, and recomputation flags. Aggregation records improved/equal/worse using tolerance `1e-12`; it does not clamp worse results or impose an invalid post-hoc monotonicity check.
