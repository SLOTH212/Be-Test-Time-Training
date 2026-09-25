"""HIT-only scale-up adapters around the byte-frozen NTP authority."""
from functools import wraps

FROZEN_CHUNK_SIZE = 1024
HIT_DEBUG_CHUNK_SIZE = 4096


def enable_debug_chunk_size(chunk_size: int) -> None:
    """Permit exactly the audited HIT debug chunk size without editing vendor code."""
    chunk_size = int(chunk_size)
    if chunk_size == FROZEN_CHUNK_SIZE:
        return
    if chunk_size != HIT_DEBUG_CHUNK_SIZE:
        raise ValueError(
            f"HIT_DEBUG_CHUNK_SIZE_NOT_AUTHORIZED requested={chunk_size} "
            f"allowed={HIT_DEBUG_CHUNK_SIZE}"
        )
    from hf_models.hf_qwen3 import ttt_state_core as core
    installed = getattr(core, "_hit_debug_chunk_override", None)
    if installed is not None:
        if installed != chunk_size:
            raise RuntimeError("CONFLICTING_HIT_DEBUG_CHUNK_OVERRIDE")
        return
    original = core.run_document

    @wraps(original)
    def debug_run_document(*args, **kwargs):
        runtime_chunk = int(kwargs.get("chunk_size", FROZEN_CHUNK_SIZE))
        if runtime_chunk != HIT_DEBUG_CHUNK_SIZE:
            raise ValueError(
                f"HIT_DEBUG_RUNTIME_CHUNK_MISMATCH expected={HIT_DEBUG_CHUNK_SIZE} "
                f"actual={runtime_chunk}"
            )
        kwargs["allow_test_override"] = True
        return original(*args, **kwargs)

    core.run_document = debug_run_document
    core._hit_debug_chunk_override = chunk_size
