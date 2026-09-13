# Mechanism definitions

Use `configs/mechanisms/<mechanism>.json` with `dynamic-ttt mechanism --config CONFIG --cohort cohort.jsonl --benchmark benchmark.jsonl --fixed fixed_scores.json --output run_dir`. Add `--plan` to inspect without loading a model. Paths and artifact SHA256 values must match the supplied model and benchmark. Cohort records require action_sequence, dynamic_score, sample_best, native_prediction_sha256, benchmark_global_index, sample_id and task. The fixed JSON maps sample IDs to complete action-to-score maps. Native prediction parity is mandatory. Public CSVs intentionally do not contain every execution trace required by this interface.

| Mechanism | Historical formal definition | Separate engineering status |
|---|---|---|
| Reverse | Reverse action order exactly; preserve prompt order, chunks, initial state and generation; replay downstream. | Llama backend implements real replay. |
| Reset | First switch normalized position <= .25; late-half modal action fraction >= .8. Reset accumulated state at the one-based terminal-run boundary while preserving prefix KV. | V2 early bound max(2,floor(N/4)) is an unfrozen candidate. |
| K/P | W held Native in all four cells. K selects control/native prefix keys and values at all attention layers. P selects all configured vs Native full-sequence used-layer union for state materialization. F11 must reproduce Native. | Do not interpret P as a learned-memory on/off switch. |
| Deletion | Reset a selected accumulated layer component at the terminal-run boundary. ALL-prefix branches evaluate components separately and report the specified mean; sham eligibility is explicit. | `experimental/llama_backend.py` omits exactly one chosen chunk/layer update, preserves that chunk's forward output, then recomputes downstream. Not equivalent to accumulated deletion. |
| R2 | Exact A!=B, 1<=tau<T family; constants and endpoint degeneracies excluded. | Eight actions with T=4 gives 168 candidates; same family convention. |

R2 tie order is score, earliest tau, configured A order, configured B order, then schedule hash, with tolerance 1e-12. Later formal R2 evidence resolves the earlier convention ambiguity for the 173-sample improved cohort (99,120 candidate outputs). Historical seven actions yield 42*(T-1) candidates per sample.

Historical Deletion and R2 definitions are resolved by their formal authorities. Whether to promote Llama omission or Reset V2 to future paper protocols remains a user decision. This snapshot does not promote either. Historical Reset has 167 eligible samples out of 173 improved; retain exclusions and denominator policies. Bootstrap estimands and zero/nonpositive-denominator handling are preserved in `dynamic_ttt.stats`.
