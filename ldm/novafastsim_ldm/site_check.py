"""Validate site.env / NOVA_FASTSIM_INPUTS before generate or submit.

Two kinds of path are involved:

- files this machine embeds in the job (card, frame configuration, a sim-only
  ``--events`` file): checked here, on this machine;
- paths at the site (``BDNMC_NOVA_SRC``, ``GEN_IMAGE``, ``NOVA_SETUP``, ``NOVA_FCL``,
  ``NOVA_IMAGE``, ``--events-at``): opaque strings, required to be non-empty only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from novafastsim import inputs, site

# Manifest roles (RELEASE.json rows carry ``role``; stub rows also carry ``"stub": true``).
# The public code names roles only, never file paths inside a private release.
ROLE_FRAME = "frame"
ROLE_SIM_CONFIG = "sim_config"
# Stub rows (identity frame, configuration marker) are NOT the detector. They are used
# only with NOVA_FASTSIM_ALLOW_STUB=1, so a forgotten BDNMC_NOVA_CONFIG cannot silently
# generate events in the wrong geometry.
MANIFEST = "RELEASE.json"
ALLOW_STUB = "NOVA_FASTSIM_ALLOW_STUB"

# Names filled from NOVA_FASTSIM_INPUTS in this process (reported by check-site / dry-runs).
FILLED: dict[str, str] = {}
# Names left unset because only a stub file was available (no NOVA_FASTSIM_ALLOW_STUB).
STUB_SKIPPED: dict[str, str] = {}


def _unset(name: str) -> bool:
    """``KEY=`` in site.env means unset, same as an absent key."""
    return not os.environ.get(name)


def verify_inputs_root(root: Path, needed: str) -> None:
    """Fail closed unless ``root/RELEASE.json`` lists ``needed`` and every listed file checks ok."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"NOVA_FASTSIM_INPUTS has no {MANIFEST}: {root}")
    rows = inputs.check(root, manifest)
    if needed not in {r["path"] for r in rows}:
        raise ValueError(f"{needed} is not listed in {manifest}")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        raise ValueError(f"NOVA_FASTSIM_INPUTS does not match {manifest}: {bad}")


def role_paths(root: Path, role: str) -> tuple[str | None, str | None]:
    """``(production, stub)`` relative paths for ``role`` in ``root/RELEASE.json``."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"NOVA_FASTSIM_INPUTS has no {MANIFEST}: {root}")
    rows = [r for r in inputs.load_manifest(manifest)["files"] if r.get("role") == role]
    prod = [r["path"] for r in rows if not r.get("stub")]
    stub = [r["path"] for r in rows if r.get("stub")]
    if len(prod) > 1 or len(stub) > 1:
        raise ValueError(f"{manifest}: more than one {'stub ' if len(stub) > 1 else ''}row with role {role!r}")
    return (prod[0] if prod else None), (stub[0] if stub else None)


def _fill(name: str, root: Path, role: str) -> None:
    if not _unset(name):
        return
    rel, stub = role_paths(root, role)
    if rel is None and stub is not None:
        if os.environ.get(ALLOW_STUB) != "1":
            # Not an error yet: an explicit --config / override may make it moot.
            STUB_SKIPPED[name] = stub
            return
        print(f"warning: {name} <- stub {stub} (not the NOvA ND; {ALLOW_STUB}=1)", file=sys.stderr)
        rel = stub
    if rel is None:
        return
    verify_inputs_root(root, rel)
    os.environ[name] = str(root / rel)
    FILLED[name] = rel


def _require(name: str) -> str:
    """``site.require`` with a pointed message when only a stub was available."""
    if _unset(name) and name in STUB_SKIPPED:
        raise ValueError(
            f"{name} is unset and NOVA_FASTSIM_INPUTS holds only the stub {STUB_SKIPPED[name]} "
            f"(not the NOvA ND). Set {name} to a real file, or {ALLOW_STUB}=1 to use the stub on purpose.")
    return site.require(name)


def apply_inputs_defaults() -> dict[str, str]:
    """If NOVA_FASTSIM_INPUTS is set, fill BDNMC_NOVA_CONFIG / NOVA_FCL when unset or empty.

    Files are found by manifest ``role`` (``frame``, ``sim_config``). Production rows are
    preferred; stub rows (``"stub": true``) need ``NOVA_FASTSIM_ALLOW_STUB=1``.
    The root is verified against its ``RELEASE.json`` before a file is taken from it.
    Returns the names filled so far in this process.
    """
    root = os.environ.get("NOVA_FASTSIM_INPUTS")
    if root:
        root_p = Path(root)
        _fill("BDNMC_NOVA_CONFIG", root_p, ROLE_FRAME)
        _fill("NOVA_FCL", root_p, ROLE_SIM_CONFIG)
    return dict(FILLED)


def require_file(label: str, path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _card_keys(card_text: str) -> set[str]:
    keys: set[str] = set()
    for line in card_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        keys.add(line.split(None, 1)[0])
    return keys


def check_card_overrides(card: Path, config: Path) -> None:
    """Fail closed if frame ``card_overrides`` name keys absent from the card.

    Private detector frames override cuboid placement (``width``/``det-theta``/…). A
    MiniBooNE-like sphere card (common in BdNMC tests) lacks those keys and makes
    ``bdnmc_nova.generate`` raise ``KeyError`` after the Slurm job starts.
    """
    cfg = json.loads(Path(config).read_text())
    overrides = cfg.get("card_overrides") or {}
    if not overrides:
        return
    missing = sorted(set(overrides) - _card_keys(Path(card).read_text()))
    if missing:
        raise ValueError(
            f"BDNMC_NOVA_CONFIG card_overrides keys not in CARD: {missing}. "
            "Use a cuboid detector card (see examples/data/generic_cuboid_pi0_m005.dat)."
        )


def _value(name: str, overrides: dict | None) -> str:
    """An explicit override wins over the environment; otherwise the variable is required."""
    value = (overrides or {}).get(name)
    return value if value else _require(name)


def _with_filled(out: dict) -> dict:
    if FILLED:
        out["from_inputs"] = dict(FILLED)
    return out


def check_generate(card: Path | None = None, config: Path | None = None) -> dict:
    """Return resolved paths for the generate stage; fail closed if missing."""
    apply_inputs_defaults()
    card_p = Path(card) if card else Path(site.require("CARD"))
    config_p = Path(config) if config else Path(_require("BDNMC_NOVA_CONFIG"))
    require_file("CARD", card_p)
    require_file("BDNMC_NOVA_CONFIG", config_p)
    check_card_overrides(card_p, config_p)
    return _with_filled({"card": str(card_p.resolve()), "config": str(config_p.resolve())})


def check_submit(card: Path | None = None, config: Path | None = None, stages: str = "gen,sim",
                 overrides: dict | None = None, events: Path | None = None,
                 events_at: str | None = None) -> dict:
    """Resolve what ``bdnmc_nova.jobspec`` needs for ``stages``; fail closed on anything missing.

    ``overrides`` maps site variable names (``NOVAFS_JOB_HANDLE``, ``BDNMC_NOVA_SRC``,
    ``GEN_IMAGE``, ``NOVA_IMAGE``, ``NOVA_SETUP``, ``NOVA_FCL``) to explicit values that win
    over site.env. Sim alone needs exactly one of ``events`` (a local HEPEVT file,
    embedded) or ``events_at`` (its path at the site).
    """
    apply_inputs_defaults()
    want = stages.split(",")
    out: dict = {"stages": stages, "handle": _value("NOVAFS_JOB_HANDLE", overrides)}
    if "gen" in want:
        if events or events_at:
            raise ValueError("--events / --events-at are for --stages sim alone (gen writes the events)")
        out.update(check_generate(card=card, config=config))
        for name in ("BDNMC_NOVA_SRC", "GEN_IMAGE"):
            out[name] = _value(name, overrides)
    if "sim" in want:
        for name in ("NOVA_SETUP", "NOVA_FCL"):
            out[name] = _value(name, overrides)
        # Optional in bdnmc_nova.jobspec: "" runs the simulation in the job's own environment.
        out["NOVA_IMAGE"] = (overrides or {}).get("NOVA_IMAGE") or os.environ.get("NOVA_IMAGE", "")
        if "gen" not in want:
            if bool(events) == bool(events_at):
                raise ValueError("--stages sim: pass --events (a local file, embedded) "
                                 "or --events-at (a path at the site)")
            if events:
                out["events"] = str(require_file("--events", Path(events)).resolve())
            else:
                out["events_at"] = events_at
    return _with_filled(out)
