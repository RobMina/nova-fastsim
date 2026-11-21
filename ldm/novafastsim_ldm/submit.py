"""Stage simulate: submit bdnmc_nova.jobspec (stock detector simulation in its own image)."""

from __future__ import annotations

import os
from pathlib import Path

from novafastsim import dtype, site

from . import deps


def _site_or(name: str, override: str | None) -> str:
    if override:
        return override
    return site.require(name)


def _optional_site(name: str, override: str | None) -> str | None:
    if override:
        return override
    value = os.environ.get(name) or ""
    return value or None


def run(card: Path | None = None, handle: str | None = None, endpoint: str | None = None,
        stages: str = "gen,sim", config: Path | None = None, src: str | None = None,
        gen_image: str | None = None, nova_image: str | None = None, nova_setup: str | None = None,
        nova_fcl: str | None = None, gpus: int | None = None, precision: str | None = None,
        trials: int | None = None, events: Path | None = None, events_at: str | None = None,
        max_events: int | None = None, dry_run_script: Path | None = None,
        extra: list[str] | None = None, dry_run: bool = False):
    """Build argv for ``bdnmc_nova.jobspec`` submit or script (dry-run).

    Gen-stage variables (``CARD``, ``BDNMC_NOVA_SRC``, ``GEN_IMAGE``, …) are required
    only when ``gen`` is in ``stages``. Sim-stage variables (``NOVA_SETUP``, ``NOVA_FCL``)
    only when ``sim`` is in ``stages``. ``NOVA_IMAGE`` is optional for sim (passed through
    when set). Sim-only jobs need ``--events`` (local embed) or ``--events-at`` (site path).
    """
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    want_gen = "gen" in stage_list
    want_sim = "sim" in stage_list
    if not want_gen and not want_sim:
        raise ValueError(f"stages must include gen and/or sim, got {stages!r}")

    handle = handle or site.require("NOVAFS_JOB_HANDLE")
    prec = dtype.resolve(precision)

    argv = ["--handle", handle, "--stages", stages, "--precision", prec]

    if want_gen:
        card_p = Path(card) if card else Path(site.require("CARD"))
        config_p = Path(config) if config else Path(site.require("BDNMC_NOVA_CONFIG"))
        argv += [
            "--config", str(config_p),
            "--src", _site_or("BDNMC_NOVA_SRC", src),
            "--gen-image", _site_or("GEN_IMAGE", gen_image),
        ]
    elif config is not None or os.environ.get("BDNMC_NOVA_CONFIG"):
        # Optional frame for sim-only when provided (jobspec still accepts --config).
        config_p = Path(config) if config else Path(site.require("BDNMC_NOVA_CONFIG"))
        argv += ["--config", str(config_p)]

    if want_sim:
        argv += [
            "--nova-setup", _site_or("NOVA_SETUP", nova_setup),
            "--nova-fcl", _site_or("NOVA_FCL", nova_fcl),
        ]
        img = _optional_site("NOVA_IMAGE", nova_image)
        if img:
            argv += ["--nova-image", img]
        if not want_gen:
            if bool(events) == bool(events_at):
                raise ValueError(
                    "--stages sim: pass --events (local file to embed) or --events-at (path at the site)"
                )
            if events is not None:
                argv += ["--events", str(events)]
            if events_at:
                argv += ["--events-at", events_at]
            if max_events is not None:
                argv += ["--max-events", str(max_events)]

    if gpus is None and os.environ.get("NOVAFS_GPUS"):
        gpus = int(site.require("NOVAFS_GPUS"))
    if gpus:
        argv += ["--gpus", str(gpus)]
    if trials is not None:
        argv += ["--trials", str(trials)]
    if extra:
        argv += list(extra)
    if want_gen:
        argv.append(str(card_p))

    if dry_run:
        return {"dry_run": True, "argv_tail": argv, "precision": prec, "stages": stages}

    deps.require_bdnmc_nova()
    from bdnmc_nova import jobspec as bn_jobspec

    if dry_run_script is not None:
        return bn_jobspec.main(["script", "--out", str(dry_run_script), *argv])

    endpoint = endpoint or site.require("NOVAFS_ENDPOINT")
    return bn_jobspec.main(["submit", "--endpoint", endpoint, *argv])
