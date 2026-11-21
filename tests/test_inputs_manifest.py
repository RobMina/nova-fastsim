"""Private-inputs manifest validate/check (no private files)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from novafastsim import inputs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "setup" / "RELEASE.example.json"


def test_validate_example_manifest():
    man = inputs.load_manifest(EXAMPLE)
    assert man["schema_version"] == 1
    assert man["release"].startswith("inputs-")
    assert len(man["files"]) >= 1


def test_validate_rejects_bad_sha(tmp_path):
    bad = json.loads(EXAMPLE.read_text())
    bad["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="sha256"):
        inputs.validate(bad)


def test_check_example_against_empty_root(tmp_path):
    rows = inputs.check(tmp_path, EXAMPLE)
    assert rows
    assert all(r["status"] == "missing" for r in rows)


def test_cli_validate():
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim.inputs", "validate", str(EXAMPLE)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ok"] is True
