"""Stage simulate: submit bdnmc_nova.jobspec (stock detector simulation in its own image)."""

from __future__ import annotations

import os
from pathlib import Path

from novafastsim import dtype, site

from . import deps, site_check


def _stages(stages: str) -> list[str]:
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    if "gen" not in stage_list and "sim" not in stage_list:
        raise ValueError(f"stages must include gen and/or sim, got {stages!r}")
    return stage_list


def run(card: Path | None = None, handle: str | None = None, endpoint: str | None = None,
        stages: str = "gen,sim", config: Path | None = None, src: str | None = None,
        gen_image: str | None = None, nova_image: str | None = None, nova_setup: str | None = None,
        nova_fcl: str | None = None, gpus: int | None = None, gpu_type: str | None = None,
        precision: str | None = None, trials: int | None = None, cores: int | None = None,
        mem_gb: int | None = None, minutes: int | None = None, preamble: str | None = None,
        events: Path | None = None, events_at: str | None = None, max_events: int | None = None,
        dry_run_script: Path | None = None, extra: list[str] | None = None, dry_run: bool = False):
    """Build argv for ``bdnmc_nova.jobspec`` submit or script (dry-run).

    Explicit arguments win over site.env. Only the variables the requested stages need
    are required: gen needs card / config / ``BDNMC_NOVA_SRC`` / ``GEN_IMAGE``; sim needs
    ``NOVA_SETUP`` / ``NOVA_FCL`` (``NOVA_IMAGE`` optional), and sim alone needs ``events``
    or ``events_at``. Sim alone passes a frame configuration through when one is given.
    ``NOVA_DEVICE_FLOAT`` / ``precision`` is validated and always passed.
    """
    want = _stages(stages)
    stages = ",".join(want)
    prec = dtype.resolve(precision)
    overrides = {
        "NOVAFS_JOB_HANDLE": handle, "BDNMC_NOVA_SRC": src, "GEN_IMAGE": gen_image,
        "NOVA_IMAGE": nova_image, "NOVA_SETUP": nova_setup, "NOVA_FCL": nova_fcl,
    }
    resolved = site_check.check_submit(card=card, config=config, stages=stages, overrides=overrides,
                                       events=events, events_at=events_at)

    argv = ["--handle", resolved["handle"], "--stages", stages, "--precision", prec]
    if "gen" in want:
        argv += ["--config", resolved["config"], "--src", resolved["BDNMC_NOVA_SRC"],
                 "--gen-image", resolved["GEN_IMAGE"]]
    elif config is not None or os.environ.get("BDNMC_NOVA_CONFIG"):
        # Optional frame for sim alone when provided (jobspec still accepts --config).
        config_p = Path(config) if config else Path(site.require("BDNMC_NOVA_CONFIG"))
        argv += ["--config", str(site_check.require_file("BDNMC_NOVA_CONFIG", config_p).resolve())]
    if "sim" in want:
        argv += ["--nova-setup", resolved["NOVA_SETUP"], "--nova-fcl", resolved["NOVA_FCL"]]
        if resolved["NOVA_IMAGE"]:
            argv += ["--nova-image", resolved["NOVA_IMAGE"]]
        if "events" in resolved:
            argv += ["--events", resolved["events"]]
            if max_events is not None and max_events >= 0:
                argv += ["--max-events", str(max_events)]
        elif "events_at" in resolved:
            argv += ["--events-at", resolved["events_at"]]

    if gpus is None and os.environ.get("NOVAFS_GPUS"):
        gpus = int(site.require("NOVAFS_GPUS"))
    if gpus is not None:
        argv += ["--gpus", str(gpus)]
    if gpu_type:
        argv += ["--gpu-type", gpu_type]
    if trials is not None:
        argv += ["--trials", str(trials)]
    if cores is not None:
        argv += ["--cores", str(cores)]
    if mem_gb is not None:
        argv += ["--mem-gb", str(mem_gb)]
    if minutes is not None:
        argv += ["--minutes", str(minutes)]
    if extra:
        argv += list(extra)
    if "gen" in want:
        argv.append(resolved["card"])

    if dry_run:
        return {"dry_run": True, "argv_tail": argv, "precision": prec, "stages": stages,
                "resolved": resolved, "preamble": preamble or ""}

    deps.require_bdnmc_nova()
    from bdnmc_nova import jobspec as bn_jobspec

    if dry_run_script is not None:
        return bn_jobspec.main(["script", "--out", str(dry_run_script), *argv])

    endpoint = endpoint or site.require("NOVAFS_ENDPOINT")
    cmd = ["submit", "--endpoint", endpoint]
    if preamble:
        cmd += ["--preamble", preamble]
    return bn_jobspec.main([*cmd, *argv])
