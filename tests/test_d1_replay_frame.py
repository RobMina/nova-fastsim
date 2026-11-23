"""Stock-deposit replay frame (benchmarks/d1/d1_replay.py) on a synthetic two-plane description."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "d1"))
R = pytest.importorskip("d1_replay")


@pytest.fixture
def g(tmp_path):
    desc = {"constants": {"cell_depth": 5.0, "cell_width_wide": 3.5, "cell_width_narrow": 3.4, "web": 0.4, "wall": 0.5,
                          "cell_length": 400.0},
            "planes": [{"z": 10.0, "view": 1, "u": [-2.0, 2.0], "narrow": [False, False], "long_centre": 1.5},
                       {"z": 17.0, "view": 0, "u": [-1.0, 3.0], "narrow": [False, False], "long_centre": -2.5}]}
    p = tmp_path / "d.json"
    p.write_text(json.dumps(desc))
    return R.description_arrays(p)


def test_local_to_world_per_view(g):
    signs = {0: (1, 1, 1), 1: (-1, 1, 1)}
    pts = R.to_world(g, np.array([0, 1]), np.array([1, 0]), np.array([0.5, 0.5]), np.array([1.0, -1.0]),
                     np.array([100.0, 100.0]), signs)
    # plane 0 (view 1: u = y, w = x): u = 2 - 0.5, w = 1.5 + 100, z = 10 + 1
    np.testing.assert_allclose(pts[0], [101.5, 1.5, 11.0])
    # plane 1 (view 0: u = x, w = y): u = -1 + 0.5, w = -2.5 + 100, z = 17 - 1
    np.testing.assert_allclose(pts[1], [-0.5, 97.5, 16.0])


def test_point_like_pieces_get_a_length_along_the_cell(g):
    f = {k: np.array(v) for k, v in dict(fls_plane=[0, 1], fls_cell=[0, 1], fls_x0=[0.2, 0.0], fls_y0=[0.0, 0.0],
                                           fls_z0=[5.0, 1.0], fls_x1=[0.2, 0.0], fls_y1=[0.0, 0.0], fls_z1=[5.0, 3.0],
                                           fls_edep=[0.001, 0.002], fls_t=[1.0, 2.0]).items()}
    p0, p1, e, t = R.segments_world(g, f, np.array([0, 1]), {0: (1, 1, 1), 1: (1, 1, 1)})
    assert np.linalg.norm(p1[0] - p0[0]) == pytest.approx(2 * R.POINT_HALF)
    np.testing.assert_allclose((p1[0] - p0[0]) / np.linalg.norm(p1[0] - p0[0]), [1.0, 0.0, 0.0])  # view 1: along x
    assert np.linalg.norm(p1[1] - p0[1]) == pytest.approx(2.0)
    np.testing.assert_allclose(e, [1.0, 2.0])
