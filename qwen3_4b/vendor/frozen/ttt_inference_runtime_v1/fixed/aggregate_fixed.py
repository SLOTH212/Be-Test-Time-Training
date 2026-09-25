#!/usr/bin/env python3
"""Generic Fixed_(N+2), Best Single, and full-action Sample Best aggregation."""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from code.runtime import TOL, atomic_json, best_single, fixed_actions, load_config, sample_best


def load_rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def aggregate(rows, layers):
    actions = fixed_actions(layers)
    by, task = defaultdict(dict), {}
    for row in rows:
        sid, action = row["sample_id"], row["action"] if "action" in row else row.get("mode")
        if action not in actions or action in by[sid]:
            raise ValueError(f"duplicate/unknown action: {sid}:{action}")
        if row.get("status") not in ("SUCCESS", "ok") or not isinstance(row.get("score"), (int, float)) or not math.isfinite(row["score"]):
            raise ValueError(f"invalid result: {sid}:{action}")
        by[sid][action] = float(row["score"])
        if sid in task and task[sid] != row["task"]:
            raise ValueError("task mismatch")
        task[sid] = row["task"]
    if not by or any(set(scores) != set(actions) for scores in by.values()):
        raise ValueError("incomplete fixed action cells")
    wide = [{"sample_id": sid, "task": task[sid], **scores} for sid, scores in by.items()]
    winner, winner_mean = best_single(wide, layers)
    per_sample = []
    for row in wide:
        best, tied = sample_best({a: row[a] for a in actions}, layers)
        per_sample.append({**row, "sample_best": best, "sample_best_actions": tied})
    means = {a: statistics.fmean(row[a] for row in wide) for a in actions}
    return {"actions": actions, "fixed_action_n": len(actions), "sample_count": len(wide), "action_means": means, "best_single_action": winner, "best_single_mean": winner_mean, "sample_best_mean": statistics.fmean(row["sample_best"] for row in per_sample), "sample_best_definition": "max over all configured fixed actions", "per_sample": per_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = aggregate(load_rows(args.input), cfg["ttt_layers"])
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, sort_keys=True))


if __name__ == "__main__":
    main()
