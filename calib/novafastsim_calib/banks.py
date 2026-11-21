"""Bank / library path scaffolding under the ``nova`` hep-simlib profile.

Phase 2 scaffolding only: names the subsystems and cache layout the ND calib
track will use. Real recorder → bank materialisation lands when hep-simlib
tables and NOvA cell/APD response exist. No site-absolute paths — use
``NOVA_LIBRARY_CACHE`` / the profile defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from novafastsim import dtype, profile


# Subsystems that ND calib libraries are expected to name (mirrors profile.SUBSYSTEMS).
BANK_SUBSYSTEMS = profile.SUBSYSTEMS


@dataclass(frozen=True)
class BankLayout:
    """Resolved on-disk layout for nova calib banks (env-driven)."""

    profile: str
    cache_dir: str
    subsystems: tuple[str, ...]
    bank_prefix: str
    dtype: str

    def to_dict(self) -> dict:
        return asdict(self)

    def subsystem_dir(self, name: str) -> Path:
        if name not in self.subsystems:
            raise ValueError(
                f"unknown subsystem {name!r}; expected one of {list(self.subsystems)}"
            )
        return Path(self.cache_dir).expanduser() / self.profile / name


def resolve_cache_dir(override: str | None = None) -> Path:
    """Return the library cache root (``NOVA_LIBRARY_CACHE`` or profile default)."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("NOVA_LIBRARY_CACHE") or ""
    if env:
        return Path(env).expanduser()
    # Match profile.NOVA_PROFILE_KW cache_dir when hep-simlib is absent.
    return Path("~/.cache/novafastsim/libraries").expanduser()


def layout(*, cache_dir: str | None = None, precision: str | None = None) -> BankLayout:
    """Build the bank layout for the current (or stand-in) nova profile."""
    p = profile.current()
    cache = resolve_cache_dir(cache_dir)
    subsystems = tuple(getattr(p, "subsystems", BANK_SUBSYSTEMS))
    return BankLayout(
        profile=getattr(p, "name", "nova"),
        cache_dir=str(cache),
        subsystems=subsystems,
        bank_prefix=str(profile.NOVA_PROFILE_KW["bank_prefix"]),
        dtype=dtype.resolve(precision),
    )


def ensure_dirs(lay: BankLayout | None = None) -> list[str]:
    """Create empty subsystem directories under the cache; return created/existing paths."""
    lay = lay or layout()
    paths: list[str] = []
    for name in lay.subsystems:
        d = lay.subsystem_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        paths.append(str(d))
    return paths


def describe() -> str:
    """Human-readable calib bank plan (no private tables required)."""
    lay = layout()
    lines = [
        "nova-fastsim calib scaffolding (hep-simlib banks under the nova profile):",
        "",
        f"profile: {lay.profile}",
        f"dtype: {lay.dtype} (NOVA_DEVICE_FLOAT)",
        f"cache: {lay.cache_dir} (NOVA_LIBRARY_CACHE or default)",
        f"bank_prefix: {lay.bank_prefix}",
        f"subsystems: {', '.join(lay.subsystems)}",
        "",
        "Pipeline (planned): recorder → banks → hepsimlib.fit / expected.",
        "Fibre / WLS tables: optional scint-fibre-fastsim (see fibre.status).",
        "Detector simulation / generators are out of scope for this package.",
    ]
    return "\n".join(lines)
