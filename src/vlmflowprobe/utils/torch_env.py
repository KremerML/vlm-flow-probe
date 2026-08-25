"""Torch environment tweaks applied before model work."""

import logging

log = logging.getLogger(__name__)

_done = False


def disable_native_triton_ops() -> None:
    """Turn off torch._native's triton-JIT'd op overrides.

    torch >= 2.10 ships triton-backed replacements for some eager ops (e.g.
    ``bmm_outer_product``) that JIT-compile C glue on first use and hard-fail
    on machines without Python dev headers. They are a speed optimization
    only; disabling them keeps the eager path in plain ATen kernels, which is
    also the closest match to the archive harness's numerics. Safe no-op on
    torch builds without ``_native``.
    """
    global _done
    if _done:
        return
    _done = True
    try:
        from torch._native import registry

        registry.deregister_op_overrides(disable_dsl_names="triton")
        log.debug("disabled torch._native triton op overrides")
    except Exception as exc:  # pragma: no cover - torch-version dependent
        log.debug("torch._native triton disable skipped: %s", exc)
