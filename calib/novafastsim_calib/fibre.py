"""Optional ``scint-fibre-fastsim`` probe for the NOvA APD / cell track.

The optional ``[fibre]`` extra (when present) pins the public package. This module
never requires it: status reports availability so calib scaffolding can proceed
without rewriting fibre optics here.
"""

from __future__ import annotations

from typing import Any


INSTALL_HINT = (
    "pip install -e 'calib[fibre]'  # or: pip install "
    "'scint-fibre-fastsim @ git+https://github.com/RobMina/scint-fibre-fastsim.git'"
)


def status() -> dict[str, Any]:
    """Whether scint-fibre-fastsim (and its NOvA cell optics and APD backend) are importable."""
    out: dict[str, Any] = {
        "available": False,
        "version": None,
        "nova_apd": False,
        "repo": "https://github.com/RobMina/scint-fibre-fastsim",
        "install": INSTALL_HINT,
        "notes": (
            "The NOvA cell, cell optics and APD backend are scint-fibre-fastsim's; "
            "novafastsim_detsim uses them for the cell response."
        ),
    }
    try:
        import scintfibrefastsim as sff
    except ImportError:
        return out

    out["available"] = True
    out["version"] = getattr(sff, "__version__", None)
    try:
        from scintfibrefastsim.backends import NovaAPDBackend  # noqa: F401

        out["nova_apd"] = True
        out["nova_apd_class"] = "scintfibrefastsim.backends.NovaAPDBackend"
    except ImportError:
        out["nova_apd"] = False
    return out
