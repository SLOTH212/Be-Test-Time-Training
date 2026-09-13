import numpy as np
def percentile(values,q):return float(np.quantile(values,q))
def formal_dynamic_bootstrap(reconstructed,included_tasks,dyn_mean,sb_mean,seed=20260828,resamples=10000):
    prereg={'included_tasks':included_tasks};BOOTSTRAP_SEED=seed;BOOTSTRAP_DRAWS=resamples
    by_task_diff = {
        task: np.asarray([float(r["dynamic_minus_samplebest"]) for r in reconstructed if r["task"] == task], dtype=np.float64)
        for task in prereg["included_tasks"]
    }
    if any(len(x) != 500 for x in by_task_diff.values()):
        raise RuntimeError("BOOTSTRAP_TASK_SIZE_MISMATCH")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for task in prereg["included_tasks"]:
        values = by_task_diff[task]
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        draws += values[indices].sum(axis=1)
    draws /= 6000.0
    bootstrap = {
        "method": "task_stratified_paired_bootstrap_full_12task_population",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "ceiling_zero_delta_samples_included": True,
        "observed_mean_delta": dyn_mean - sb_mean,
        "bootstrap_mean_delta": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1)),
        "ci95_lower": percentile(draws, 0.025),
        "ci95_upper": percentile(draws, 0.975),
        "probability_delta_gt_zero": float(np.mean(draws > 0)),
    }
    return bootstrap
