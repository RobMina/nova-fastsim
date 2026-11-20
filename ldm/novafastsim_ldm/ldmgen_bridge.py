"""Optional ldmgen hop via the public library API (not CLI argv).

Public package: https://github.com/RobMina/ldmgen (pin via novafastsim-ldm[ldmgen]).
``export_nova`` still sinks through ``bdnmc_nova.textfilegen``.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import deps


def convert_bdnmc_jax(hdf5: Path, out: Path, detector: str = "nova-nd", **kwargs):
    """BdNMC JAX HDF5 → ``ldmgen/1`` record via ``ldmgen.adapters.bdnmc.from_jax_hdf5``."""
    deps.require_ldmgen()
    from ldmgen.adapters import bdnmc

    rec = bdnmc.from_jax_hdf5(Path(hdf5), detector=detector, **kwargs)
    from ldmgen import SCHEMA

    path = rec.save(Path(out))
    y_sc, y_dm = rec.yield_per_pot()
    return {
        "path": str(path),
        "n_dm": int(rec.n_dm),
        "n_scatter": int(rec.n_scatter),
        "signal_per_pot": float(y_sc),
        "dm_per_pot": float(y_dm),
        "schema": SCHEMA,
    }


def export_nova(record: Path, out: Path, config: Path | None = None, max_events: int = -1,
                summary: Path | None = None):
    """``ldmgen.export.to_nova`` → TextFileGen file (bdnmc_nova.textfilegen)."""
    deps.require_ldmgen()
    deps.require_bdnmc_nova()  # to_nova imports bdnmc_nova.textfilegen
    from ldmgen import EventRecord
    from ldmgen.export import to_nova, write_summary

    rec = EventRecord.load(Path(record))
    info = to_nova(rec, Path(out), config_path=config, max_events=max_events)
    if summary is not None:
        write_summary(info, Path(summary))
    return info


def info(record: Path) -> dict:
    """Load an ``ldmgen/1`` record and return counts / yields (JSON-serialisable)."""
    deps.require_ldmgen()
    from ldmgen import EventRecord

    rec = EventRecord.load(Path(record))
    y_sc, y_dm = rec.yield_per_pot()
    meta = json.loads(rec.run.to_json())
    meta.pop("card", None)
    return {
        "path": str(record),
        "generator": rec.run.generator,
        "detector": rec.run.detector,
        "n_dm": int(rec.n_dm),
        "n_scatter": int(rec.n_scatter),
        "signal_per_pot": float(y_sc),
        "dm_per_pot": float(y_dm),
        "run": meta,
    }


def status() -> dict:
    """Whether ldmgen (and bdnmc-nova for export) can be used from this environment."""
    available = deps.ldmgen_available()
    out = {
        "ldmgen_available": available,
        "bdnmc_nova_available": False,
        "repo": "https://github.com/RobMina/ldmgen",
        "install": "pip install 'novafastsim-ldm[ldmgen]'",
        "api": "ldmgen.adapters.bdnmc / ldmgen.export.to_nova",
    }
    if available:
        import ldmgen
        out["ldmgen_version"] = getattr(ldmgen, "__version__", "unknown")
        out["note"] = "ldmgen importable; export_nova still sinks through bdnmc_nova.textfilegen"
    else:
        out["note"] = "not installed in this env; " + deps.LDMGEN_INSTALL
    try:
        import bdnmc_nova  # noqa: F401
        out["bdnmc_nova_available"] = True
    except ImportError:
        pass
    return out
