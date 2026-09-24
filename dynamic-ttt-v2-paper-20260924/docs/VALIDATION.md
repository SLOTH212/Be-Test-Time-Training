# Validation boundaries

Original V2: repeatability4/4, zero-write64/64, prefix invariance15/15, apply-then-update5/5. Writable checkpoint digest / fresh-state checks: precheck144/144, Stage-1 1384/1384, Stage-2 11480/11480. Original zero-write traces cover all five candidate outputs for full chunks, tail and generation. See `results/writer/TABLE_V2_EXECUTOR_VALIDATION.tsv`.

The final adapter source hash differs. Its functional forward is AST-identical; final bridges and winner parity are separately recorded. No full original-precheck rerun on that exact final hash was found. No formal V2 alternate-gold/scorer-isolation intervention result was found. These are NOT_FOUND, not failed tests or inferred PASS.

Mechanism K1 recorded trace closure is165/165; K0 trace closure110/165 and prediction closure118/165. These are the recorded current sites, not complete historical L1–L6 coverage. Mechanism and R2 are outcome-conditioned on170 current Dynamic improvements.

Random postflight is PASS with62772 planned/evaluated, zero missing, duplicate executions, errors or unauthorized retries. The included CPU checker independently matches candidate/plan identities and final best scores; it does not claim to recreate GPU execution ledgers or model behavior.
