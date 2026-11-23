"""Tabulated geometry (per-plane z, view and long-axis centre; per-cell centre and width): equivalence with the
uniform geometry, exact cases on a staggered layout, validation, dtypes, gradients, and the private description
when one is configured. Synthetic tables only: no detector numbers in this file."""

import os
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
from novafastsim_detsim import config, geometry, particles, sim  # noqa: E402

UNI = ds.Geometry(n_planes=12, n_cells=24)


def tables_of(g: ds.Geometry) -> ds.Geometry:
    """The uniform geometry written as tables."""
    P, C = g.n_planes, g.n_cells
    z = [g.z_front + p * g.plane_pitch + g.face + 0.5 * g.cell_depth for p in range(P)]
    u = [-g.half_width + (c + 0.5) * g.cell_pitch for c in range(C)]
    return replace(g, plane_z=tuple(z), plane_view=tuple(g.view(p) for p in range(P)), cell_u=(tuple(u),) * P,
                   cell_w=((g.cell_width,) * C,) * P, long_centre=(0.0,) * P, length=g.cell_length)


def staggered(g: ds.Geometry, shift=1.3, narrow=0.9, w_off=4.0) -> ds.Geometry:
    """Odd planes shifted by ``shift``, edge cells narrower, a long-axis offset alternating by view."""
    t = tables_of(g)
    P, C = g.n_planes, g.n_cells
    cu = tuple(tuple(u + (shift if p % 2 else 0.0) for u in t.cell_u[p]) for p in range(P))
    cw = tuple(tuple(g.cell_width * (narrow if c in (0, C - 1) else 1.0) for c in range(C)) for _ in range(P))
    lc = tuple(w_off if g.view(p) == 0 else -w_off for p in range(P))
    return replace(t, cell_u=cu, cell_w=cw, long_centre=lc)


def _per_cell(det, pc):
    g = det.geometry
    out = np.zeros(g.n_planes * g.n_cells)
    np.add.at(out, np.asarray(pc.index).ravel(), np.asarray(pc.length).ravel())
    return out.reshape(g.n_planes, g.n_cells)


def _segments(det, n=4000, seed=5):
    g = det.geometry
    rng = np.random.default_rng(seed)
    p0 = np.c_[rng.uniform(-35, 35, n), rng.uniform(-35, 35, n), rng.uniform(g.z_front_edge + 5, g.z_back - 5, n)]
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return p0, p0 + d * rng.uniform(0.5, det.settings.max_segment, n)[:, None]


def test_uniform_tables_reproduce_the_uniform_geometry():
    a, b = ds.Detector(UNI), ds.Detector(tables_of(UNI))
    assert b.geometry.tabulated and b.plane_window == a.plane_window and b.cell_window == a.cell_window
    p0, p1 = _segments(a)
    pa = geometry.pieces(a, jnp.asarray(p0), jnp.asarray(p1))
    pb = geometry.pieces(b, jnp.asarray(p0), jnp.asarray(p1))
    np.testing.assert_allclose(_per_cell(b, pb), _per_cell(a, pa), atol=1e-9)
    np.testing.assert_allclose(np.asarray(pb.distance)[np.asarray(pb.length) > 0],
                               np.asarray(pa.distance)[np.asarray(pa.length) > 0], atol=1e-9)
    ua, za, va = geometry.cell_centres(a, jnp.float64)
    ub, zb, vb = geometry.cell_centres(b, jnp.float64)
    np.testing.assert_allclose(ub, ua, atol=1e-9)
    np.testing.assert_allclose(zb, za, atol=1e-9)
    assert np.array_equal(np.asarray(vb), np.asarray(va))


def test_staggered_axial_track_hits_the_tabulated_cell():
    det = ds.Detector(staggered(UNI))
    g = det.geometry
    x, y = 3.1, -6.7
    z = np.arange(g.z_front_edge - 1.0, g.z_back + 1.0, 1.0)
    p0 = np.stack([np.full(len(z) - 1, x), np.full(len(z) - 1, y), z[:-1]], 1)
    got = _per_cell(det, geometry.pieces(det, jnp.asarray(p0), jnp.asarray(p0 + [0.0, 0.0, 1.0])))
    for p in range(g.n_planes):
        u = x if g.view(p) == 0 else y
        lo = np.array(g.cell_u[p]) - 0.5 * np.array(g.cell_w[p])
        inside = (u >= lo) & (u <= lo + np.array(g.cell_w[p]))
        want = np.where(inside, g.cell_depth, 0.0)
        np.testing.assert_allclose(got[p], want, atol=1e-9)


def test_staggered_distance_uses_the_plane_long_centre():
    det = ds.Detector(staggered(UNI))
    g = det.geometry
    p = 0                                                       # view of plane 0; cells along the other axis
    z = g.plane_z[p]
    w = 2.0
    pts = [[1.0, w, z]] if g.view(p) == 0 else [[w, 1.0, z]]
    a = np.array(pts) - [[0.0, 0.0, 1.0]]
    pc = geometry.pieces(det, jnp.asarray(a), jnp.asarray(a + [[0.0, 0.0, 2.0]]))
    dist = np.asarray(pc.distance)[np.asarray(pc.length) > 0]
    sign = g.readout_end[g.view(p)]
    assert dist[0] == pytest.approx(0.5 * g.length - sign * (w - g.long_centre[p]), abs=1e-9)


def test_staggered_random_segments_close_on_the_volume_fraction():
    det = ds.Detector(staggered(UNI, shift=0.0, narrow=1.0, w_off=0.0))   # tables with uniform content
    p0, p1 = _segments(det, n=20000, seed=3)
    pc = geometry.pieces(det, jnp.asarray(p0), jnp.asarray(p1))
    assert not np.asarray(pc.overflow).any()
    frac = np.asarray(pc.length).sum() / np.asarray(pc.len_box).sum()
    assert frac == pytest.approx(det.geometry.scint_fraction, rel=0.01)


def test_incomplete_or_unsorted_tables_fail_closed():
    t = tables_of(UNI)
    with pytest.raises(ValueError):
        replace(UNI, plane_z=t.plane_z)
    with pytest.raises(ValueError):
        replace(t, plane_z=tuple(reversed(t.plane_z)))
    with pytest.raises(ValueError):
        replace(t, cell_u=t.cell_u[:-1])


def test_tables_from_description_round_trip():
    t = staggered(UNI)
    desc = {"constants": {"cell_depth": t.cell_depth, "cell_width_wide": t.cell_width,
                          "cell_width_narrow": t.cell_width * 0.9, "web": t.web, "wall": t.face,
                          "cell_length": t.length},
            "readout_end": list(t.readout_end),
            "materials": {"Scintillator": {"density_g_cm3": 0.9}, "PVC": {"density_g_cm3": 1.4, "radlen_cm": 17.0}},
            "planes": [{"z": t.plane_z[p], "view": t.plane_view[p], "u": list(t.cell_u[p]),
                        "narrow": [c in (0, t.n_cells - 1) for c in range(t.n_cells)], "long_centre": t.long_centre[p]}
                       for p in range(t.n_planes)]}
    det, params = config.from_dict(config.tables_from_description(desc))
    g = det.geometry
    assert g.tabulated and g.plane_z == t.plane_z and g.cell_u == t.cell_u and g.long_centre == t.long_centre
    np.testing.assert_allclose(np.array(g.cell_w), np.array(t.cell_w), atol=1e-12)
    assert params["rho_scint"] == 0.9 and params["rho_inert"] == 1.4
    _, p_no_radlen = config.from_dict({"materials": {"scintillator": {"density": 0.9}, "inert": {"density": 1.4}}})
    assert params["x0"] != p_no_radlen["x0"]                            # the PVC radiation length is used


def test_tabulated_response_float32_matches_float64_and_has_gradients():
    det = ds.Detector(staggered(UNI))
    params = config.default_params(det.geometry)

    def total_pe(vz, dt):
        p = config.to_device(params, dt)
        part = particles.make([13], [[0.0, 0.0, 600.0]], [[1.5, -2.0, 0.0]], dtype=dt)
        part = part._replace(vertex=part.vertex.at[0, 2].set(vz))
        return sim.expected(det, p, part).pe.sum()

    z0 = float(det.geometry.plane_z[1]) + 0.37
    a = float(total_pe(jnp.float64(z0), jnp.float64))
    b = float(total_pe(jnp.float32(z0), jnp.float32))
    assert b == pytest.approx(a, rel=1e-4)
    g = float(jax.grad(lambda v: total_pe(v, jnp.float64))(jnp.float64(z0)))
    h = 1e-4
    fd = (float(total_pe(jnp.float64(z0 + h), jnp.float64)) - float(total_pe(jnp.float64(z0 - h), jnp.float64))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-4, abs=1e-6)


@pytest.mark.skipif(not os.environ.get("NOVA_FASTSIM_INPUTS"), reason="private inputs not configured")
def test_private_description_loads_and_contains_a_shower():
    root = os.environ["NOVA_FASTSIM_INPUTS"]
    path = os.path.join(root, "nova", "detector", "nd_S26-07-01.json")
    if not os.path.isfile(path):
        pytest.skip("no ND description in the private inputs")
    det, params = config.load_description(path)
    p = config.to_device(params, jnp.float64)
    part = particles.make([11], [[1.0, 2.0, 300.0]], [[0.0, 0.05, 1000.0]], dtype=jnp.float64)
    d = sim.expected(det, p, part).diag
    assert float(d["overflow"]) == 0
    assert float(d["e_scint"] + d["e_inert"]) == pytest.approx(float(d["e_total"]) - float(d["e_outside"]), rel=1e-9)
