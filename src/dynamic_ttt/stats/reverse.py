import numpy as np
import statistics
from collections import defaultdict
CELLS=["F00","F10","F01","F11"]
LEVELS=["L1","L2","L3","L4","L5","L6"]

def mean(xs):
    return statistics.fmean(xs)


def ci(values, seed, resamples):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        n = min(1000, resamples - start)
        means[start:start+n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def stratified_ci(rows, seed, resamples):
    groups = defaultdict(list)
    for r in rows:
        groups[r["task"]].append(float(r["dynamic_minus_reverse"]))
    rng = np.random.default_rng(seed)
    out = np.zeros(resamples, dtype=np.float64)
    total = sum(len(v) for v in groups.values())
    for vals in groups.values():
        x = np.asarray(vals, dtype=np.float64)
        for start in range(0, resamples, 1000):
            n = min(1000, resamples - start)
            out[start:start+n] += x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1) * len(x) / total
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]
