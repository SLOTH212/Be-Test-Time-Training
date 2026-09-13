

def independent_winner(records):
    """Separate implementation from the worker's online selector."""
    best = max(float(x["score"]) for x in records)
    action_rank = dict(zip(ACTIONS, range(len(ACTIONS))))
    tied = []
    for x in records:
        if math.isclose(float(x["score"]), best, rel_tol=0.0, abs_tol=TOL): tied.append(x)
    selected = min(tied, key=lambda x: (int(x["tau"]), action_rank[x["A"]], action_rank[x["B"]], x["full_schedule_hash"]))
    return selected, best, len(tied)
