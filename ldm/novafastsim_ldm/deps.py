"""Optional external packages: fail with an install hint, never vendor them."""

from __future__ import annotations

LDMGEN_INSTALL = (
    "pip install 'novafastsim-ldm[ldmgen]' "
    "(pins https://github.com/RobMina/ldmgen)"
)


def require_bdnmc_nova():
    try:
        import bdnmc_nova  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "bdnmc-nova is required for generate/submit. "
            "Install: pip install 'novafastsim-ldm[chains]' (commit-pinned RobMina/bdnmc-nova)"
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
        f"ldmgen is not installed. {LDMGEN_INSTALL}. "
        "Until then use `python -m novafastsim_ldm generate` (bdnmc-nova HEPEVT path)."
    )
