"""Fixed/Sample Best/Dynamic compatible action selection; no search algorithm."""
from contextlib import contextmanager
from ntp_core.validation import validate_config

def action_layers(layers):
    if not layers or list(layers)!=sorted(set(layers)) or any(type(i) is not int or i<0 for i in layers):
        raise ValueError('layers must be nonempty, unique, sorted, nonnegative integers')
    return {'OFF':[],**{f'L{i}':[i] for i in layers},'ALL':list(layers)}

def fresh_cache(model):
    """Sample reset discards BOTH KV and TTT state, as in fresh generate calls."""
    module=__import__(type(model).__module__,fromlist=['TTTDynamicCache'])
    return module.TTTDynamicCache(config=model.config)

@contextmanager
def action_scope(model, action):
    validate_config(model.config,model.config.model_type)
    selected=set(action_layers(model.config.ttt_layers)[action])
    stashed={};previous=model.model.ttt_mode
    try:
        for i in model.config.ttt_layers:
            if i in selected:continue
            mlp=model.model.layers[i].mlp;stashed[i]={}
            for name in ('ttt_conv','ttt_proj','ttt_ntp_gate'):
                if hasattr(mlp,name):stashed[i][name]=getattr(mlp,name);delattr(mlp,name)
        if not selected:model.model.ttt_mode=False
        yield
    finally:
        model.model.ttt_mode=previous
        for i,attrs in stashed.items():
            for name,value in attrs.items():setattr(model.model.layers[i].mlp,name,value)
