"""Inject the family builder into common pipeline code without forking its recipe."""
from contextlib import contextmanager
from functools import partial
from .model_family import build

@contextmanager
def bind_family_builder(formal_module, *, model_family, ttt_layers, chunk_size=1024, **placement):
    """Use around a common caller of formal.build(path).

    This does not bypass run-specific model-scale/data-identity guards in frozen
    historical launchers. Future formal launch configuration must supply its own
    verified Llama asset/identity and the existing distributed runtime.
    """
    old_build,old_layers=formal_module.build,formal_module.LAYERS
    formal_module.build=partial(build,model_family=model_family,ttt_layers=ttt_layers,chunk_size=chunk_size,**placement)
    formal_module.LAYERS=list(ttt_layers)
    try:yield formal_module
    finally:formal_module.build,formal_module.LAYERS=old_build,old_layers
