

def terminal_run_boundary(sequence):
    action = sequence[-1]
    i = len(sequence) - 1
    while i > 0 and sequence[i - 1] == action:
        i -= 1
    if i == 0:
        raise ValueError("NO_ACTION_SWITCH")
    return i + 1


def historical_early_steady(sequence):
    """Exact executed morphology() logic from analysis_common.py."""
    n = len(sequence)
    switches = [i for i in range(1, n) if sequence[i] != sequence[i - 1]]
    first_norm = (switches[0] + 1) / n if switches and n else None
    early = first_norm is not None and first_norm <= 0.25
    tail = sequence[n // 2:]
    counts = Counter(tail)
    steady = max(ACTIONS, key=lambda a: (counts[a], -ACTIONS.index(a))) if tail else "UNKNOWN"
    steady_n = counts[steady] if tail else 0
    steady_fraction = steady_n / max(len(tail), 1)
    steady_ok = steady_fraction >= 0.8
    return bool(early and steady_ok), {
        "first_switch_chunk_index": switches[0] + 1 if switches else None,
        "first_switch_position_normalized": first_norm,
        "early_switch_at_or_before_fraction": 0.25,
        "tail_start_chunk_index": n // 2 + 1,
        "tail_length": len(tail),
        "steady_tail_action": steady,
        "steady_tail_count": steady_n,
        "steady_tail_fraction": steady_fraction,
        "steady_tail_required_fraction": 0.8,
    }
