"""Stage simulate: submit bdnmc_nova.jobspec (stock TextFileGen in SL7)."""

from __future__ import annotations

import os
from pathlib import Path

from novafastsim import site

from . import deps


def _site_or(name: str, override: str | None) -> str:
    if override:
        return override
    return site.require(name)


def run(card: Path | None = None, handle: str | None = None, endpoint: str | None = None,
        stages: str = "gen,sim", config: Path | None = None, src: str | None = None,
        gen_image: str | None = None, nova_image: str | None = None, nova_setup: str | None = None,
        nova_fcl: str | None = None, gpus: int | None = None, precision: str | None = None,
        trials: int | None = None, dry_run_script: Path | None = None, extra: list[str] | None = None):
    """Build argv for ``bdnmc_nova.jobspec`` submit or script (dry-run)."""
    deps.require_bdnmc_nova()
    from bdnmc_nova import jobspec as bn_jobspec

    card_p = Path(card) if card else Path(site.require("CARD"))
    handle = handle or site.require("NOVAFS_JOB_HANDLE")
    config_p = Path(config) if config else Path(site.require("BDNMC_NOVA_CONFIG"))

    argv = [
        "--handle", handle,
        "--stages", stages,
        "--config", str(config_p),
        "--src", _site_or("BDNMC_NOVA_SRC", src),
        "--gen-image", _site_or("GEN_IMAGE", gen_image),
        "--nova-image", _site_or("NOVA_IMAGE", nova_image),
        "--nova-setup", _site_or("NOVA_SETUP", nova_setup),
        "--nova-fcl", _site_or("NOVA_FCL", nova_fcl),
    ]
    if gpus is None and os.environ.get("NOVAFS_GPUS"):
        gpus = int(site.require("NOVAFS_GPUS"))
    if gpus:
        argv += ["--gpus", str(gpus)]
    prec = precision or os.environ.get("NOVA_DEVICE_FLOAT")
    if prec:
        argv += ["--precision", prec]
    if trials is not None:
        argv += ["--trials", str(trials)]
    if extra:
        argv += list(extra)
    argv.append(str(card_p))

    if dry_run_script is not None:
        return bn_jobspec.main(["script", "--out", str(dry_run_script), *argv])

    endpoint = endpoint or site.require("NOVAFS_ENDPOINT")
    return bn_jobspec.main(["submit", "--endpoint", endpoint, *argv])
