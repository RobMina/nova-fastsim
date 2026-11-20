"""Install the ``nova`` hep-simlib profile (or a local stand-in when hep-simlib is absent)."""

from __future__ import annotations

from dataclasses import dataclass

# Subsystems the NOvA ND track expects to name in manifests / banks.
SUBSYSTEMS = ("nd", "cell", "apd", "beamline", "component")

NOVA_PROFILE_KW = dict(
    name="nova",
    subsystems=SUBSYSTEMS,
    dtype_env="NOVA_DEVICE_FLOAT",
    default_dtype="float64",
    cache_env="NOVA_LIBRARY_CACHE",
    cache_dir="~/.cache/novafastsim/libraries",
    bank_subsystem="component",
    bank_prefix="component.",
)


@dataclass(frozen=True)
class LocalProfile:
    """Minimal stand-in matching hepsimlib.profile.Profile fields used here."""

    name: str = "nova"
    subsystems: tuple = SUBSYSTEMS
    dtype_env: str = "NOVA_DEVICE_FLOAT"
    default_dtype: str = "float64"


_INSTALLED = None


def install():
    """Make the nova profile current. Prefer hepsimlib when installed."""
    global _INSTALLED
    try:
        from hepsimlib import profile as hp

        p = hp.Profile(**NOVA_PROFILE_KW)
        hp.configure(p)
        _INSTALLED = p
        return p
    except ImportError:
        _INSTALLED = LocalProfile()
        return _INSTALLED


def current():
    """Return the installed profile, installing on first use."""
    if _INSTALLED is None:
        return install()
    return _INSTALLED
