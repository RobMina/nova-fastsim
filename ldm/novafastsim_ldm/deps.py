"""Optional external packages: fail with an install hint, never vendor them."""

from __future__ import annotations


def require_bdnmc_nova():
    try:
        import bdnmc_nova  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "bdnmc-nova is required for generate/submit. "
            "Install: pip install 'novafastsim-ldm[chains]' "
            "or pip install 'bdnmc-nova @ git+https://github.com/RobMina/bdnmc-nova.git'"
        ) from e


def ldmgen_available() -> bool:
    try:
        import ldmgen  # noqa: F401
        return True
    except ImportError:
        return False


def require_ldmgen():
    if ldmgen_available():
        return True
    raise ImportError(
        "ldmgen is not installed. It is not on GitHub yet (local ~/ldm-generators). "
        "When published, install it and re-run; until then use "
        "`python -m novafastsim_ldm generate` (bdnmc-nova HEPEVT path)."
    )
