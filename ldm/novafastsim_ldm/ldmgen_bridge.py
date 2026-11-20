"""Optional ldmgen hop: convert / export nova when ldmgen is installed.

Public GitHub for ldmgen is not ready yet; until then this module reports that
clearly and the HEPEVT path stays on bdnmc_nova.generate.
"""

from __future__ import annotations

from pathlib import Path

from . import deps


def convert_bdnmc_jax(hdf5: Path, out: Path, **kwargs):
    """``ldmgen convert bdnmc-jax <hdf5> --out <record.h5>``."""
    deps.require_ldmgen()
    from ldmgen import cli as ldm_cli

    argv = ["convert", "bdnmc-jax", str(hdf5), "--out", str(out)]
    for k, v in kwargs.items():
        if v is None:
            continue
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return ldm_cli.main(argv)


def export_nova(record: Path, out: Path, config: Path | None = None):
    """``ldmgen export nova <record> --out events.txt`` (still uses bdnmc_nova.textfilegen)."""
    deps.require_ldmgen()
    from ldmgen import cli as ldm_cli

    argv = ["export", "nova", str(record), "--out", str(out)]
    if config is not None:
        argv += ["--config", str(config)]
    return ldm_cli.main(argv)


def status() -> dict:
    """Whether ldmgen can be used from this environment."""
    return {
        "ldmgen_available": deps.ldmgen_available(),
        "note": ("install local ~/ldm-generators or the future public package"
                 if not deps.ldmgen_available()
                 else "ldmgen importable; export nova still sinks through bdnmc_nova.textfilegen"),
    }
