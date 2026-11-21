"""NOvA ND JAX fast-simulation core (profile, dtype, site, inputs).

Submodules are loaded lazily via ``__getattr__`` so ``python -m novafastsim.inputs``
does not eagerly import ``dtype`` (and so a bad ``NOVA_DEVICE_FLOAT`` does not break
unrelated tools).
"""

from __future__ import annotations

__all__ = ["dtype", "inputs", "profile", "site"]
__version__ = "0.0.2"

_LAZY = frozenset(__all__)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
