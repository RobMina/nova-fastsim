"""Device floating-point mode via NOVA_DEVICE_FLOAT (float32 | float64). Fail closed."""

from __future__ import annotations

import os

import numpy as np

DEFAULT = "float64"
NAME = os.environ.get("NOVA_DEVICE_FLOAT", DEFAULT) or DEFAULT
if NAME not in ("float32", "float64"):
    raise ValueError(f"NOVA_DEVICE_FLOAT must be float32 or float64, not {NAME!r}")

FLOAT = np.dtype(NAME)
SINGLE = FLOAT == np.float32


def compute_dtype(name: str | None = None) -> np.dtype:
    """Return the device float dtype; ``name`` overrides the environment for one call."""
    if name is None:
        return FLOAT
    if name not in ("float32", "float64"):
        raise ValueError(f"device float must be float32 or float64, not {name!r}")
    return np.dtype(name)
