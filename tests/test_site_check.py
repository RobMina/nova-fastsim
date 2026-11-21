"""site_check + NOVA_FASTSIM_INPUTS defaults + submit argv per stage (no bdnmc-nova needed)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from novafastsim import site
from novafastsim_ldm import site_check, submit

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "tests" / "data" / "identity_frame.json"
CARD = ROOT / "examples" / "data" / "generic_cuboid_pi0_m005.dat"

SITE_VARS = (
    "NOVA_FASTSIM_INPUTS", "NOVA_FASTSIM_ALLOW_STUB", "BDNMC_NOVA_CONFIG", "NOVA_FCL", "CARD",
    "BDNMC_NOVA_SRC", "GEN_IMAGE", "NOVA_IMAGE", "NOVA_SETUP", "NOVAFS_JOB_HANDLE", "NOVAFS_GPUS",
    "NOVAFS_ENDPOINT", "NOVA_DEVICE_FLOAT",
)


@pytest.fixture(autouse=True)
def clean_site(monkeypatch):
    for name in SITE_VARS:
        monkeypatch.delenv(name, raising=False)
    site_check.FILLED.clear()
    site_check.STUB_SKIPPED.clear()
    yield
    site_check.FILLED.clear()
    site_check.STUB_SKIPPED.clear()


# Test-only relative paths: any layout works because files are found by manifest role.
FRAME = "frame/frame.json"
SIM_CONFIG = "config/sim.cfg"
FRAME_STUB = "frame/identity_frame.json"
SIM_CONFIG_STUB = "config/sim.STUB"


def _release(root: Path, files: dict[str, tuple[str, str, bool]]) -> Path:
    """Write ``files`` (path -> (text, role, stub)) under ``root`` plus a matching RELEASE.json."""
    rows = []
    for rel, (text, role, stub) in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        data = path.read_bytes()
        row = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
               "provenance": "test", "redistribution": "nova-internal", "role": role}
        if stub:
            row["stub"] = True
        rows.append(row)
    (root / "RELEASE.json").write_text(json.dumps(
        {"schema_version": 1, "release": "inputs-test", "files": rows}))
    return root


def _stub_release(root: Path) -> Path:
    return _release(root, {FRAME_STUB: (IDENTITY.read_text(), site_check.ROLE_FRAME, True),
                           SIM_CONFIG_STUB: ("# stub\n", site_check.ROLE_SIM_CONFIG, True)})


def _prod_release(root: Path) -> Path:
    return _release(root, {FRAME: (IDENTITY.read_text(), site_check.ROLE_FRAME, False),
                           SIM_CONFIG: ("#include \"x.fcl\"\n", site_check.ROLE_SIM_CONFIG, False)})


# --- NOVA_FASTSIM_INPUTS defaults -------------------------------------------------------

def test_inputs_defaults_fill_production_files(tmp_path, monkeypatch):
    _prod_release(tmp_path)
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    filled = site_check.apply_inputs_defaults()
    assert Path(os.environ["BDNMC_NOVA_CONFIG"]) == tmp_path / FRAME
    assert Path(os.environ["NOVA_FCL"]) == tmp_path / SIM_CONFIG
    assert filled == {"BDNMC_NOVA_CONFIG": FRAME, "NOVA_FCL": SIM_CONFIG}


def test_empty_values_from_site_env_do_not_block_defaults(tmp_path, monkeypatch):
    """A site.env copied from site.env.example has ``BDNMC_NOVA_CONFIG=`` / ``NOVA_FCL=``."""
    inputs_root = _prod_release(tmp_path / "inputs")
    env = tmp_path / "site.env"
    env.write_text(f"BDNMC_NOVA_CONFIG=\nNOVA_FCL=\nNOVA_FASTSIM_INPUTS={inputs_root}\n")
    site.load_dotenv(env)  # clean_site already removed these names (and restores them after)
    site_check.apply_inputs_defaults()
    assert Path(os.environ["BDNMC_NOVA_CONFIG"]) == inputs_root / FRAME
    assert Path(os.environ["NOVA_FCL"]) == inputs_root / SIM_CONFIG


def test_stub_not_used_without_opt_in(tmp_path, monkeypatch):
    _stub_release(tmp_path)
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    monkeypatch.setenv("CARD", str(CARD))
    assert site_check.apply_inputs_defaults() == {}
    assert "BDNMC_NOVA_CONFIG" not in os.environ
    with pytest.raises(ValueError, match="stub"):
        site_check.check_generate()


def test_stub_skipped_but_explicit_config_still_works(tmp_path, monkeypatch):
    _stub_release(tmp_path)
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    got = site_check.check_generate(card=CARD, config=IDENTITY)
    assert Path(got["config"]) == IDENTITY.resolve()


def test_stub_used_with_opt_in(tmp_path, monkeypatch, capsys):
    _stub_release(tmp_path)
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    monkeypatch.setenv("NOVA_FASTSIM_ALLOW_STUB", "1")
    site_check.apply_inputs_defaults()
    assert Path(os.environ["BDNMC_NOVA_CONFIG"]) == tmp_path / FRAME_STUB
    assert Path(os.environ["NOVA_FCL"]) == tmp_path / SIM_CONFIG_STUB
    assert "not the NOvA ND" in capsys.readouterr().err


def test_inputs_root_must_match_manifest(tmp_path, monkeypatch):
    _prod_release(tmp_path)
    (tmp_path / FRAME).write_text("{}")  # changed after the release was cut
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    with pytest.raises(ValueError, match="does not match"):
        site_check.apply_inputs_defaults()


def test_inputs_root_needs_manifest(tmp_path, monkeypatch):
    frame = tmp_path / FRAME
    frame.parent.mkdir(parents=True)
    frame.write_text(IDENTITY.read_text())
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="RELEASE.json"):
        site_check.apply_inputs_defaults()


# --- generate checks ---------------------------------------------------------------------

def test_check_generate_ok(monkeypatch):
    monkeypatch.setenv("CARD", str(CARD))
    monkeypatch.setenv("BDNMC_NOVA_CONFIG", str(IDENTITY))
    got = site_check.check_generate()
    assert Path(got["card"]).is_file()
    assert Path(got["config"]).is_file()


def test_check_generate_missing(monkeypatch):
    monkeypatch.setenv("CARD", "/no/such/card.dat")
    monkeypatch.setenv("BDNMC_NOVA_CONFIG", str(IDENTITY))
    with pytest.raises(FileNotFoundError, match="CARD"):
        site_check.check_generate()


def test_card_overrides_require_cuboid_keys(tmp_path):
    card = tmp_path / "sphere.dat"
    card.write_text("detector sphere\nradius 5.0\nx-position 0.0\n")
    cfg = tmp_path / "frame.json"
    cfg.write_text('{"frame": {}, "card_overrides": {"width": "3.9", "det-theta": "0.01"}}\n')
    with pytest.raises(ValueError, match="width"):
        site_check.check_card_overrides(card, cfg)


def test_card_overrides_ok_for_cuboid(tmp_path):
    cfg = tmp_path / "frame.json"
    cfg.write_text('{"frame": {}, "card_overrides": {"width": "3.9", "det-theta": "0.01"}}\n')
    site_check.check_card_overrides(CARD, cfg)


# --- submit argv per stage ----------------------------------------------------------------

def _gen_env(monkeypatch):
    monkeypatch.setenv("CARD", str(CARD))
    monkeypatch.setenv("BDNMC_NOVA_CONFIG", str(IDENTITY))
    # Site paths are opaque strings: never checked on this machine.
    monkeypatch.setenv("BDNMC_NOVA_SRC", "/site/src/bdnmc-nova")
    monkeypatch.setenv("GEN_IMAGE", "/site/containers/bdnmc-jax.sif")
    monkeypatch.setenv("NOVAFS_JOB_HANDLE", "env-handle")


def _sim_env(monkeypatch):
    monkeypatch.setenv("NOVA_SETUP", "source /site/setup.sh")
    monkeypatch.setenv("NOVA_FCL", "/site/fcl/sim.fcl")
    monkeypatch.setenv("NOVAFS_JOB_HANDLE", "env-handle")


def test_submit_gen_only_needs_no_sim_variables(monkeypatch):
    _gen_env(monkeypatch)
    out = submit.run(stages="gen", gpus=0, dry_run=True)
    argv = out["argv_tail"]
    assert not any(a.startswith("--nova-") for a in argv)
    assert argv[argv.index("--gen-image") + 1] == "/site/containers/bdnmc-jax.sif"
    assert argv[argv.index("--gpus") + 1] == "0"
    assert argv[-1] == str(CARD.resolve())


def test_submit_sim_only_needs_no_gen_variables(monkeypatch):
    _sim_env(monkeypatch)
    out = submit.run(stages="sim", events_at="/site/runs/h/out/events.txt", dry_run=True)
    argv = out["argv_tail"]
    for flag in ("--config", "--src", "--gen-image", "--nova-image"):
        assert flag not in argv  # NOVA_IMAGE unset -> run in the job's own environment
    assert argv[argv.index("--events-at") + 1] == "/site/runs/h/out/events.txt"
    assert argv[argv.index("--nova-fcl") + 1] == "/site/fcl/sim.fcl"


def test_submit_sim_only_embeds_local_events(tmp_path, monkeypatch):
    _sim_env(monkeypatch)
    monkeypatch.setenv("NOVA_IMAGE", "/site/sl7.sif")
    events = tmp_path / "events.txt"
    events.write_text("0 0\n")
    argv = submit.run(stages="sim", events=events, max_events=10, dry_run=True)["argv_tail"]
    assert argv[argv.index("--events") + 1] == str(events.resolve())
    assert argv[argv.index("--max-events") + 1] == "10"
    assert argv[argv.index("--nova-image") + 1] == "/site/sl7.sif"


def test_submit_sim_only_needs_exactly_one_events_source(tmp_path, monkeypatch):
    _sim_env(monkeypatch)
    with pytest.raises(ValueError, match="--events"):
        submit.run(stages="sim", dry_run=True)
    with pytest.raises(ValueError, match="--events"):
        submit.run(stages="sim", events=tmp_path / "e.txt", events_at="/site/e.txt", dry_run=True)


def test_submit_rejects_events_with_gen(monkeypatch):
    _gen_env(monkeypatch)
    with pytest.raises(ValueError, match="sim alone"):
        submit.run(stages="gen", events_at="/site/e.txt", dry_run=True)


def test_check_site_and_submit_agree_on_nova_image(monkeypatch):
    _sim_env(monkeypatch)
    checked = site_check.check_submit(stages="sim", events_at="/site/e.txt")
    assert checked["NOVA_IMAGE"] == ""
    submit.run(stages="sim", events_at="/site/e.txt", dry_run=True)  # must not raise either


def test_explicit_overrides_win_over_site_env(monkeypatch):
    _gen_env(monkeypatch)
    monkeypatch.setenv("GEN_IMAGE", "")  # e.g. ``GEN_IMAGE=`` left empty in site.env
    out = submit.run(stages="gen", handle="cli-handle", gen_image="/site/other.sif", dry_run=True)
    argv = out["argv_tail"]
    assert argv[argv.index("--handle") + 1] == "cli-handle"
    assert argv[argv.index("--gen-image") + 1] == "/site/other.sif"
    assert out["resolved"]["GEN_IMAGE"] == "/site/other.sif"


def test_manifest_role_must_be_unique(tmp_path, monkeypatch):
    _release(tmp_path, {FRAME: (IDENTITY.read_text(), site_check.ROLE_FRAME, False),
                        "frame/other.json": (IDENTITY.read_text(), site_check.ROLE_FRAME, False)})
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    with pytest.raises(ValueError, match="more than one"):
        site_check.apply_inputs_defaults()


def test_release_without_role_fills_nothing(tmp_path, monkeypatch):
    _release(tmp_path, {FRAME: (IDENTITY.read_text(), "notes", False)})
    monkeypatch.setenv("NOVA_FASTSIM_INPUTS", str(tmp_path))
    assert site_check.apply_inputs_defaults() == {}
