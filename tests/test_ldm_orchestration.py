"""Phase-2 orchestration: describe, ldmgen status, fail-closed deps (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_describe_cli():
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim_ldm", "describe"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "ldm") + ":" + str(ROOT / "core")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "TextFileGen" in proc.stdout
    assert "bdnmc-nova" in proc.stdout


def test_ldmgen_status_without_package():
    from novafastsim_ldm import ldmgen_bridge

    st = ldmgen_bridge.status()
    assert "ldmgen_available" in st
    assert isinstance(st["ldmgen_available"], bool)


def test_require_bdnmc_nova_message():
    from novafastsim_ldm import deps

    # If already installed in the env, just assert the helper returns True.
    try:
        assert deps.require_bdnmc_nova() is True
    except ImportError as e:
        assert "bdnmc-nova" in str(e)


def test_chain_path_owners():
    from novafastsim_ldm import chain

    owners = {p.stage: p.owner for p in chain.PATH}
    assert "generate" in owners
    assert "simulate" in owners
    assert "TextFileGen" in owners["simulate"]


def test_hepsimbench_stub_status():
    script = ROOT / "benchmarks" / "hepsimbench_nd.py"
    proc = subprocess.run(
        [sys.executable, str(script), "status"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "available" in data
