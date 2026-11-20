"""Optional ldmgen hop: convert / export nova when ldmgen is installed.

Public package: https://github.com/RobMina/ldmgen (pin via novafastsim-ldm[ldmgen]).
``export nova`` still sinks through bdnmc_nova.textfilegen.
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
    available = deps.ldmgen_available()
    return {
        "ldmgen_available": available,
        "repo": "https://github.com/RobMina/ldmgen",
        "install": "pip install 'novafastsim-ldm[ldmgen]'",
        "note": (
            "ldmgen importable; export nova still sinks through bdnmc_nova.textfilegen"
            if available
            else "not installed in this env; " + deps.LDMGEN_INSTALL
        ),
    }
