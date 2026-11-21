"""submit.run argv construction per stage (dry-run; no bdnmc-nova)."""

from __future__ import annotations

from pathlib import Path

import pytest

from novafastsim_ldm import submit

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "tests" / "data" / "identity_frame.json"  # path stand-in
CFG = CARD


def _gen_env(monkeypatch):
    monkeypatch.setenv("CARD", str(CARD))
    monkeypatch.setenv("BDNMC_NOVA_CONFIG", str(CFG))
    monkeypatch.setenv("BDNMC_NOVA_SRC", "/project/example/src/bdnmc-nova")
    monkeypatch.setenv("GEN_IMAGE", "/project/example/containers/gen.sif")
    monkeypatch.setenv("NOVAFS_JOB_HANDLE", "h-gen")
    monkeypatch.delenv("NOVA_DEVICE_FLOAT", raising=False)


def _sim_env(monkeypatch):
    monkeypatch.setenv("NOVA_SETUP", "/opt/setup.sh")
    monkeypatch.setenv("NOVA_FCL", "/opt/sim.fcl")
    monkeypatch.setenv("NOVAFS_JOB_HANDLE", "h-sim")
    monkeypatch.delenv("NOVA_DEVICE_FLOAT", raising=False)
    monkeypatch.delenv("NOVA_IMAGE", raising=False)


def test_gen_only_argv(monkeypatch):
    _gen_env(monkeypatch)
    got = submit.run(stages="gen", dry_run=True)
    argv = got["argv_tail"]
    assert "--stages" in argv and "gen" in argv
    assert "--precision" in argv
    assert argv[argv.index("--precision") + 1] == "float64"  # nova-fastsim default
    assert "--gen-image" in argv
    assert "--nova-setup" not in argv
    assert str(CARD) in argv
    assert "--events" not in argv


def test_sim_only_needs_events(monkeypatch):
    _sim_env(monkeypatch)
    with pytest.raises(ValueError, match="events"):
        submit.run(stages="sim", dry_run=True)


def test_sim_only_events_at(monkeypatch):
    _sim_env(monkeypatch)
    got = submit.run(stages="sim", events_at="$HEPEP_SLURM_WORKDIR/prev/out/events.txt", dry_run=True)
    argv = got["argv_tail"]
    assert "--stages" in argv and "sim" in argv
    assert "--events-at" in argv
    assert "--nova-setup" in argv
    assert "--nova-fcl" in argv
    assert "--gen-image" not in argv
    assert "--src" not in argv
    # card path must not appear as a trailing positional
    assert not any(str(CARD) == a for a in argv)


def test_sim_only_does_not_require_gen_vars(monkeypatch):
    _sim_env(monkeypatch)
    for name in ("CARD", "BDNMC_NOVA_CONFIG", "BDNMC_NOVA_SRC", "GEN_IMAGE"):
        monkeypatch.delenv(name, raising=False)
    got = submit.run(stages="sim", events_at="/site/events.txt", dry_run=True)
    assert got["stages"] == "sim"


def test_sim_nova_image_optional(monkeypatch):
    _sim_env(monkeypatch)
    got = submit.run(stages="sim", events_at="/site/events.txt", dry_run=True)
    assert "--nova-image" not in got["argv_tail"]
    monkeypatch.setenv("NOVA_IMAGE", "/opt/nova.sif")
    got2 = submit.run(stages="sim", events_at="/site/events.txt", dry_run=True)
    assert "--nova-image" in got2["argv_tail"]


def test_gen_sim_requires_both(monkeypatch):
    _gen_env(monkeypatch)
    _sim_env(monkeypatch)
    monkeypatch.setenv("NOVAFS_JOB_HANDLE", "h-both")
    got = submit.run(stages="gen,sim", dry_run=True)
    argv = got["argv_tail"]
    assert "--gen-image" in argv and "--nova-setup" in argv
    assert str(CARD) in argv


def test_bad_precision_fails_closed(monkeypatch):
    _gen_env(monkeypatch)
    monkeypatch.setenv("NOVA_DEVICE_FLOAT", "float16")
    with pytest.raises(ValueError, match="float16"):
        submit.run(stages="gen", dry_run=True)
