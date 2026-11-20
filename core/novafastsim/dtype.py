"""Device floating-point mode via ``NOVA_DEVICE_FLOAT`` (float32 | float64). Fail closed.

Resolution is **lazy**: call :func:`resolve` / :func:`compute_dtype` after ``site.env``
is loaded. Importing this module never reads the environment (so a bad value does
not break ``python -m novafastsim.inputs``).
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT = "float64"
ALLOWED = ("float32", "float64")


def resolve(name: str | None = None) -> str:
    """Return ``float32`` or ``float64``; unset → :data:`DEFAULT`; other values raise."""
    if name is None:
        raw = os.environ.get("NOVA_DEVICE_FLOAT") or DEFAULT
    else:
        raw = name
    if raw not in ALLOWED:
        raise ValueError(f"NOVA_DEVICE_FLOAT must be float32 or float64, not {raw!r}")
    return raw


def compute_dtype(name: str | None = None) -> np.dtype:
    """Return the device float dtype; ``name`` overrides the environment for one call."""
    return np.dtype(resolve(name))


def apply_jax_x64(name: str | None = None) -> str:
    """Map resolved dtype → ``jax.config.update(\"jax_enable_x64\", …)`` when JAX is importable.

    Returns the resolved name. No-op (still validating) if JAX is not installed.
    """
    resolved = resolve(name)
    try:
        import jax
    except ImportError:
        return resolved
    jax.config.update("jax_enable_x64", resolved == "float64")
    return resolved


def __getattr__(name: str):
    """Lazy ``FLOAT`` / ``SINGLE`` / ``NAME`` for callers that still expect module constants."""
    if name == "NAME":
        return resolve()
    if name == "FLOAT":
        return compute_dtype()
    if name == "SINGLE":
        return compute_dtype() == np.dtype("float32")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
