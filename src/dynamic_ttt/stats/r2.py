"""Frozen paired and task-stratified R2 bootstrap; ratios use cohort sums."""
import numpy as np
from collections import defaultdict

def ci(values): return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap(data, seed, n_resamples, stratified=False):
    rng = np.random.default_rng(seed); gaps = []; ratios = []
    groups = defaultdict(list)
    for i, x in enumerate(data): groups[x["task"]].append(i)
    all_indices = np.arange(len(data))
    for _ in range(n_resamples):
        if stratified:
            idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for _, v in sorted(groups.items())])
        else: idx = rng.choice(all_indices, size=len(data), replace=True)
        gaps.append(float(np.mean([data[i]["Dynamic_minus_R2"] for i in idx])))
        den = sum(data[i]["Dynamic_minus_OFF"] for i in idx)
        ratios.append(sum(data[i]["R2_minus_OFF"] for i in idx) / den)
    return {"mean_dynamic_minus_r2_95ci": ci(gaps), "aggregate_r2_gain_retention_95ci": ci(ratios),
            "seed": seed, "resamples": n_resamples, "stratified_by_task": stratified}
