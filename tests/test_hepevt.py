"""HEPEVT input helpers (orchestration only; no detector simulation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from novafastsim_ldm import hepevt

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """# comment kept by head
0 2
1 11 0 -1 -1 -1 0.1 0.2 0.3 0.5 0.000511 1.0 2.0 3.0 0.0
2 18 0 -1 -1 -1 0.0 0.0 1.0 2.0 0.1 1.0 2.0 3.0 0.0
1 1
1 22 0 -1 -1 -1 0.0 0.0 0.5 0.5 0.0 0.0 0.0 10.0 1.0
"""


def test_validate_and_count():
    info = hepevt.validate(SAMPLE)
    assert info.n_events == 2
    assert info.n_particles == 3
    assert info.events[0] == hepevt.EventSummary(index=0, n_particles=2)
    assert info.events[1] == hepevt.EventSummary(index=1, n_particles=1)
    assert hepevt.count_events(SAMPLE) == 2


def test_malformed_header():
    with pytest.raises(ValueError, match="event header"):
        hepevt.validate("not a header\n")


def test_truncated_block():
    with pytest.raises(ValueError, match="remain"):
        hepevt.validate("0 2\n1 11 0 -1 -1 -1 0 0 0 1 0 0 0 0 0\n")


def test_head_text_keeps_leading_comment(tmp_path):
    out = hepevt.head_text(SAMPLE, 1)
    assert out.startswith("# comment")
    assert hepevt.count_events(out) == 1
    assert "1 22" not in out


def test_inspect_file(tmp_path):
    p = tmp_path / "events.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    info = hepevt.inspect(p, max_events=1)
    assert info.n_events == 2
    assert len(info.events) == 1
    assert info.path == str(p)
    assert info.bytes == p.stat().st_size


def test_events_info_cli(tmp_path):
    p = tmp_path / "events.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(ROOT / "ldm") + ":" + str(ROOT / "core"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim_ldm", "events-info", str(p), "--max-list", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["n_events"] == 2
    assert data["n_particles"] == 3
    assert len(data["events"]) == 1


def test_events_head_cli(tmp_path):
    src = tmp_path / "events.txt"
    dst = tmp_path / "head.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(ROOT / "ldm") + ":" + str(ROOT / "core"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "novafastsim_ldm", "events-head", str(src), "--n", "1", "--out", str(dst)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert hepevt.count_events(dst.read_text(encoding="utf-8")) == 1
