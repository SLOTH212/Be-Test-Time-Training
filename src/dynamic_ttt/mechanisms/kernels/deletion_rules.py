

def category(prefix, next_action):
    written = set().union(*(ACTION_LAYERS[a] for a in prefix)) if prefix else set()
    post = set(ACTION_LAYERS[next_action])
    if "ALL" in prefix:
        return "ALL_PREFIX"
    if "OFF" in prefix or next_action == "OFF":
        return "OFF_INVOLVING"
    if len(written) == 1 and len(post) == 1 and written.isdisjoint(post):
        return "SINGLE_LAYER_DISJOINT"
    if written & post:
        return "OVERLAPPING_LAYER"
    return "COMPLEX_PREFIX"


def first_real_layer(prefix):
    for action in prefix:
        layers = ACTION_LAYERS[action]
        if action == "ALL":
            raise RuntimeError("ALL_PREFIX_MUST_USE_LOO")
        if layers:
            return layers[0]
    return None
