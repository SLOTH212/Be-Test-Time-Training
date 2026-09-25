"""Candidate morphology screening only. No execution of interventions."""
from collections import Counter
def screen(sequence, actions):
    n=len(sequence)
    if not n or any(a not in actions for a in sequence):
        raise ValueError("EMPTY_OR_UNKNOWN_ACTION")
    switches=[i for i in range(1,n) if sequence[i]!=sequence[i-1]]
    s=switches[0]+1 if switches else None
    limit=max(2,n//4)
    tail=sequence[n//2:]
    counts=Counter(tail)
    modal=max(actions,key=lambda a:(counts[a],-actions.index(a)))
    fraction=counts[modal]/max(len(tail),1)
    return {"eligible":s is not None and s<=limit and fraction>=0.8,
            "N":n,"first_switch":s,"first_fraction":s/n if s else None,
            "early_limit":limit,"late_length":len(tail),"late_modal":modal,
            "late_count":counts[modal],"late_fraction":fraction}
def complete_chunks(token_count,chunk_size):
    if token_count<0 or chunk_size<=0: raise ValueError("INVALID_GEOMETRY")
    return token_count//chunk_size
