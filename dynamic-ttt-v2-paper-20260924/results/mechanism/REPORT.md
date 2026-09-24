# V2 mechanism replication (P omitted)

Core complete: 170/170; R2 complete: 170/170.

| Experiment | N | Native % | Condition % | Drop pp | 95% CI |
|---|---:|---:|---:|---:|---|
| reverse | 170 | 93.5294 | 41.3235 | 52.2059 | [50.0, 54.294117647058826] |
| reset | 165 | 93.3333 | 46.6263 | 46.7071 | [43.35353535353536, 49.90909090909091] |
| nontrivial_reset | 157 | 93.2484 | 44.1614 | 49.0870 | [46.12526539278131, 51.825902335456476] |
| deletion | 157 | 93.2484 | 54.3079 | 38.9406 | [35.167728237791934, 42.579883227176204] |
| sham | 109 | 93.7003 | 93.7003 | 0.0000 | [0.0, 0.0] |
| K0 | 165 | 93.3333 | 78.7778 | 14.5556 | [10.535353535353536, 18.767676767676768] |
| K1 | 165 | 93.3333 | 93.3333 | 0.0000 | [0.0, 0.0] |
| K1_minus_K0 | 165 | 93.3333 | 78.7778 | 14.5556 | [10.535353535353536, 18.767676767676768] |
| r2 | 170 | 93.5294 | 87.1863 | 6.3431 | [3.4117647058823533, 9.607843137254902] |

Selection is conditional on current Dynamic improvement; sample sizes differ from historical cohorts. R2 is hindsight reference-scored exhaustive one-switch search. K closure uses explicitly recorded trace hashes, not an unmeasured claim of full historical L1-L6 coverage.

{
  "n": 165,
  "K0_trace_exact": 110,
  "K1_trace_exact": 165,
  "K0_prediction_exact": 118
}
{
  "n": 170,
  "candidates": 97356,
  "reused": 7733,
  "at_least_native": 154,
  "exact_native": 154,
  "native_one_switch": 92,
  "native_multi_switch": 78,
  "multi_switch_at_least_native": 62
}
