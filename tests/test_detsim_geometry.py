"""Segment-to-cell path lengths: exact cases, volume-fraction closure, window overflow."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
from novafastsim_detsim import geometry  # noqa: E402

DET = ds.Detector(ds.Geometry(n_planes=12, n_cells=24))


def _per_cell(pc):
    g = DET.geometry
    out = np.zeros(g.n_planes * g.n_cells)
    np.add.at(out, np.asarray(pc.index).ravel(), np.asarray(pc.length).ravel())
    return out.reshape(g.n_planes, g.n_cells)


def test_axial_track_hits_one_cell_per_plane():
    g = DET.geometry
    i, j = 5, 17
    x = -g.half_width + (i + 0.5) * g.cell_pitch
    y = -g.half_width + (j + 0.5) * g.cell_pitch
    z = np.arange(g.z_front - 1.0, g.z_back + 1.0, 1.0)
    p0 = np.stack([np.full(len(z) - 1, x), np.full(len(z) - 1, y), z[:-1]], 1)
    p1 = p0 + [0.0, 0.0, 1.0]
    lengths = _per_cell(geometry.pieces(DET, jnp.asarray(p0), jnp.asarray(p1)))
    for p in range(g.n_planes):
        c = i if g.view(p) == 0 else j
        assert lengths[p, c] == pytest.approx(g.cell_depth, abs=1e-9)
        assert lengths[p].sum() == pytest.approx(g.cell_depth, abs=1e-9)


def test_transverse_segment_matches_cell_intervals():
    g = DET.geometry
    z = g.z_front + g.face + 0.5 * g.cell_depth          # plane 0, x view
    a, b = -7.3, 5.1
    pc = geometry.pieces(DET, jnp.array([[a, 3.0, z]]), jnp.array([[b, 3.0, z]]))
    got = _per_cell(pc)[0]
    lo = -g.half_width + np.arange(g.n_cells) * g.cell_pitch + 0.5 * g.web
    want = np.clip(np.minimum(lo + g.cell_width, b) - np.maximum(lo, a), 0.0, None)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_random_segments_close_on_the_scintillator_fraction():
    g = DET.geometry
    rng = np.random.default_rng(3)
    S = 20000
    p0 = np.c_[rng.uniform(-40, 40, S), rng.uniform(-40, 40, S), rng.uniform(10, g.z_back - 10, S)]
    d = rng.normal(size=(S, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    p1 = p0 + d * rng.uniform(0.5, DET.settings.max_segment, S)[:, None]
    pc = geometry.pieces(DET, jnp.asarray(p0), jnp.asarray(p1))
    assert not np.asarray(pc.overflow).any()
    frac = np.asarray(pc.length).sum() / np.asarray(pc.len_box).sum()
    assert frac == pytest.approx(g.scint_fraction, rel=0.01)


def test_long_segment_is_flagged():
    pc = geometry.pieces(DET, jnp.array([[0.0, 0.0, 10.0]]), jnp.array([[30.0, 0.0, 60.0]]))
    assert bool(pc.overflow[0])


def test_segment_outside_is_empty():
    pc = geometry.pieces(DET, jnp.array([[500.0, 0.0, 10.0]]), jnp.array([[505.0, 0.0, 12.0]]))
    assert float(pc.len_box[0]) == 0.0 and float(jnp.sum(pc.length)) == 0.0
