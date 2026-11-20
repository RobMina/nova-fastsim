"""ldmgen bridge uses the library API; smoke with a toy ldmgen/1 record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ldmgen")

from ldmgen import EventRecord, Model, RunMeta  # noqa: E402
from novafastsim_ldm import ldmgen_bridge  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def _toy(n=5, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.5, 5, n)
    th = rng.uniform(0, 0.02, n)
    p4 = np.column_stack([e, e * np.sin(th), np.zeros(n), e * np.cos(th)])
    run = RunMeta(
        generator="toy", generator_version="0", pot=1e20, dm_sample="scattered",
        model=Model("vector_portal", 1e-3, 0.1, 0.005, 0.1, "scalar"), detector="test",
        card="# toy card\ndark_matter_mass 0.005\n",
    )
    w = np.full(n, 1.0)
    dm = {"event": np.arange(n), "p4": p4, "vertex": np.zeros((n, 4)), "parent_pdg": 111,
          "parent_kind": 1, "weight": w}
    el = p4 * 0.3
    el[:, 0] = np.linalg.norm(el[:, 1:], axis=1)
    sc = {
        "event": np.arange(n), "dm_index": np.arange(n),
        "vertex": np.column_stack([np.zeros((n, 2)), rng.uniform(490, 500, n), np.full(n, 1.6e-6)]),
        "target_pdg": np.full(n, 11), "out_pdg": np.full(n, 11),
        "out_p4": el, "out_mass": 0.000511, "chi_out_p4": p4 - el, "weight": w,
    }
    return EventRecord.build(run, dm, sc)


def test_status_reports_api():
    st = ldmgen_bridge.status()
    assert st["ldmgen_available"] is True
    assert "ldmgen.export.to_nova" in st["api"]
    assert st["repo"].endswith("/ldmgen")


def test_info_on_toy_record(tmp_path):
    path = _toy().save(tmp_path / "toy.h5")
    info = ldmgen_bridge.info(path)
    assert info["n_dm"] == 5
    assert info["n_scatter"] == 5
    assert info["generator"] == "toy"


def test_export_nova_with_identity_frame(tmp_path):
    pytest.importorskip("bdnmc_nova")
    rec = _toy().save(tmp_path / "toy.h5")
    out = tmp_path / "events.txt"
    summary = tmp_path / "export.json"
    info = ldmgen_bridge.export_nova(
        rec, out, config=DATA / "identity_frame.json", summary=summary)
    assert out.is_file() and out.stat().st_size > 0
    assert info["events"] == 5
    assert json.loads(summary.read_text())["events"] == 5
