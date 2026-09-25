"""Same weighted Stage2 objective; release each graph before the next forward."""
def backward_terms(model, ids, answer_mask, context_mask, answer_weight, context_weight, fused_sum):
    answer = fused_sum(model, ids, answer_mask) * answer_weight
    value = answer.detach().double()
    answer.backward()
    del answer
    context = fused_sum(model, ids, context_mask) * context_weight
    value = value + context.detach().double()
    context.backward()
    del context
    return value
