"""Calib scaffolding: bank layout + optional fibre probe (no hep-simlib required)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _env(**extra):
    path = str(ROOT / "calib") + ":" + str(ROOT / "core")
    return {**os.environ, "PYTHONPATH": path, **extra}


def test_banks_layout_uses_cache_env(tmp_path, monkeypatch):
    from novafastsim_calib import banks

    monkeypatch.setenv("NOVA_LIBRARY_CACHE", str(tmp_path / "libs"))
    monkeypatch.setenv("NOVA_DEVICE_FLOAT", "float32")
    lay = banks.layout()
    assert lay.profile == "nova"
    assert lay.dtype == "float32"
    assert lay.cache_dir == str(tmp_path / "libs")
    assert "cell" in lay.subsystems
    assert "apd" in lay.subsystems
    cell = lay.subsystem_dir("cell")
    assert cell == tmp_path / "libs" / "nova" / "cell"


def test_banks_unknown_subsystem(tmp_path, monkeypatch):
    from novafastsim_calib import banks

    monkeypatch.setenv("NOVA_LIBRARY_CACHE", str(tmp_path))
    lay = banks.layout()
    with pytest.raises(ValueError, match="unknown subsystem"):
        lay.subsystem_dir("not-a-subsystem")


def test_ensure_dirs(tmp_path, monkeypatch):
    from novafastsim_calib import banks

    monkeypatch.setenv("NOVA_LIBRARY_CACHE", str(tmp_path / "cache"))
    created = banks.ensure_dirs()
    assert len(created) == len(banks.BANK_SUBSYSTEMS)
    for p in created:
        assert Path(p).is_dir()


def test_fibre_status_without_package():
    from novafastsim_calib import fibre

    st = fibre.status()
    assert "available" in st
    assert st["repo"].endswith("scint-fibre-fastsim")
    assert "pip" in st["install"]


def test_fibre_status_with_pinned_package():
    pytest.importorskip("scintfibrefastsim")
    from novafastsim_calib import fibre

    st = fibre.status()
    assert st["available"] is True
    assert st["version"]
    assert st["nova_apd"] is True


def test_describe_cli():
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim_calib", "describe"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert "hep-simlib" in proc.stdout
    assert "NOVA_LIBRARY_CACHE" in proc.stdout


def test_status_cli(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim_calib", "status", "--cache-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(NOVA_DEVICE_FLOAT="float64"),
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["banks"]["profile"] == "nova"
    assert data["banks"]["cache_dir"] == str(tmp_path)
    assert "fibre" in data
    assert "available" in data["fibre"]
