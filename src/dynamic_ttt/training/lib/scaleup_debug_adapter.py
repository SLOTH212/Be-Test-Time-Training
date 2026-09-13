"""SCALEUP-only scale-up adapters around the byte-frozen NTP authority."""
from functools import wraps

FROZEN_CHUNK_SIZE = 1024
SCALEUP_DEBUG_CHUNK_SIZE = 4096


def enable_debug_chunk_size(chunk_size: int) -> None:
    """Permit exactly the audited SCALEUP debug chunk size without editing vendor code."""
    chunk_size = int(chunk_size)
    if chunk_size == FROZEN_CHUNK_SIZE:
        return
    if chunk_size != SCALEUP_DEBUG_CHUNK_SIZE:
        raise ValueError(
            f"SCALEUP_DEBUG_CHUNK_SIZE_NOT_AUTHORIZED requested={chunk_size} "
            f"allowed={SCALEUP_DEBUG_CHUNK_SIZE}"
        )
    from dynamic_ttt.ttt import qwen_state as core
    installed = getattr(core, "_scaleup_debug_chunk_override", None)
    if installed is not None:
        if installed != chunk_size:
            raise RuntimeError("CONFLICTING_SCALEUP_DEBUG_CHUNK_OVERRIDE")
        return
    original = core.run_document

    @wraps(original)
    def debug_run_document(*args, **kwargs):
        runtime_chunk = int(kwargs.get("chunk_size", FROZEN_CHUNK_SIZE))
        if runtime_chunk != SCALEUP_DEBUG_CHUNK_SIZE:
            raise ValueError(
                f"SCALEUP_DEBUG_RUNTIME_CHUNK_MISMATCH expected={SCALEUP_DEBUG_CHUNK_SIZE} "
                f"actual={runtime_chunk}"
            )
        kwargs["allow_test_override"] = True
        return original(*args, **kwargs)

    core.run_document = debug_run_document
    core._scaleup_debug_chunk_override = chunk_size
