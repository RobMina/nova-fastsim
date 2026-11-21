"""Stage generate: BdNMC JAX → HEPEVT text file via bdnmc_nova.generate."""

from __future__ import annotations

import os
from pathlib import Path

from novafastsim import dtype, site

from . import deps, site_check


def _out_path(out: Path | None) -> Path:
    if out is not None:
        return Path(out)
    if os.environ.get("NOVAFS_EVENTS_OUT"):
        return Path(site.require("NOVAFS_EVENTS_OUT"))
    return Path("events.txt")


def run(card: Path | None = None, out: Path | None = None, config: Path | None = None,
        hdf5: Path | None = None, summary: Path | None = None, trials: int | None = None,
        channel: str | None = None, extra: list[str] | None = None, dry_run: bool = False):
    """Call ``bdnmc_nova.generate`` with site.env defaults for unset paths.

    Validates ``NOVA_DEVICE_FLOAT`` (via :func:`novafastsim.dtype.resolve`) and the card /
    frame files (via :func:`site_check.check_generate`) before building argv. Precision is
    recorded in the dry-run payload; ``bdnmc_nova.generate`` itself does not take a
    ``--precision`` flag today — JAX dtype is applied via
    :func:`novafastsim.dtype.apply_jax_x64` when generation actually runs.
    """
    prec = dtype.resolve()
    resolved = site_check.check_generate(card=card, config=config)
    card_p = Path(resolved["card"])
    config_p = Path(resolved["config"])
    out_p = _out_path(out)
    argv = [str(card_p), "--out", str(out_p), "--config", str(config_p)]
    if trials is not None:
        argv += ["--trials", str(trials)]
    if channel:
        argv += ["--channel", channel]
    if hdf5:
        argv += ["--hdf5", str(hdf5)]
    if summary:
        argv += ["--summary", str(summary)]
    if extra:
        argv += list(extra)
    if dry_run:
        return {"dry_run": True, "argv": argv, **resolved, "precision": prec, "out": str(out_p)}

    dtype.apply_jax_x64(prec)
    deps.require_bdnmc_nova()
    from bdnmc_nova import generate as bn_generate
    return bn_generate.main(argv)
