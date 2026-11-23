"""Shower-shape extras (electron head, shower-maximum shift, lateral scales, shower inert stopping) and materials."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
testing = pytest.importorskip("hepsimlib.testing")
from novafastsim_detsim import config, reco  # noqa: E402

DET = ds.Detector(ds.Geometry(n_planes=128, n_cells=96))
BASE = ds.default_params(DET.geometry)


def run(pdg, over=None, mom=(0.0, 0.0, 2000.0), vtx=(1.0, 0.7, 30.0)):
    p = ds.to_device(BASE, "float64")
    p.update({k: jnp.asarray(v, jnp.float64) for k, v in (over or {}).items()})
    return ds.expected(DET, p, ds.make([pdg], [list(vtx)], [list(mom)], dtype=jnp.float64))


def first_x0_fraction(r):
    per_plane = np.asarray(r.edep).sum(1)
    n = int(np.ceil(BASE["x0"] / DET.geometry.plane_pitch))
    return per_plane[:n + 5].sum() / per_plane.sum()   # vertex plane is about plane 4


def test_neutral_extras_are_the_original_model():
    a = run(11)
    b = run(11, {"head_x0": 0.0, "shower_tmax_shift": 0.0, "lat_core": 1.0, "lat_tail": 1.0, "lat_pcore": 1.0,
                 "shower_inert_scale": 1.0})
    np.testing.assert_array_equal(np.asarray(a.edep), np.asarray(b.edep))


def test_head_closes_energy_and_raises_the_start():
    r0, r1 = run(11), run(11, {"head_x0": 1.5})
    d = {k: float(v) for k, v in r1.diag.items()}
    T = np.hypot(2000.0, 0.51099895) - 0.51099895
    assert d["e_total"] == pytest.approx(T, rel=1e-9)
    assert d["e_scint"] + d["e_inert"] == pytest.approx(T, rel=1e-6)
    assert first_x0_fraction(r1) > 1.2 * first_x0_fraction(r0)
    g0, g1 = run(22), run(22, {"head_x0": 1.5})                       # photons have no head
    np.testing.assert_allclose(np.asarray(g1.edep), np.asarray(g0.edep), rtol=1e-12, atol=1e-12)


def test_tmax_shift_moves_the_shower_deeper():
    def depth(r):
        w = np.asarray(r.edep).sum(1)
        return (w * np.arange(len(w))).sum() / w.sum()
    shift = 1.0
    dz = (depth(run(11, {"shower_tmax_shift": shift})) - depth(run(11))) * DET.geometry.plane_pitch
    assert dz == pytest.approx(shift * BASE["x0"], rel=0.15)


def central_fraction(r):
    e = np.asarray(r.edep)
    views = np.arange(DET.geometry.n_planes) % 2 == 0
    per_cell = e[views].sum(0)
    return per_cell.max() / per_cell.sum()


def test_lateral_scales_move_energy_in_and_out_of_the_core():
    base = central_fraction(run(11))
    assert central_fraction(run(11, {"lat_core": 0.5})) > base
    assert central_fraction(run(11, {"lat_tail": 2.0})) < base
    assert central_fraction(run(11, {"lat_pcore": 1.3})) > base


def test_shower_inert_scale_moves_shower_energy_to_inert_only():
    e0, e1 = run(11), run(11, {"shower_inert_scale": 1.3})
    assert float(e1.diag["e_scint"]) < float(e0.diag["e_scint"])
    assert float(e1.diag["e_scint"] + e1.diag["e_inert"]) == pytest.approx(float(e1.diag["e_total"]), rel=1e-6)
    m0 = run(13, {}, mom=(0.0, 0.0, 600.0))
    m1 = run(13, {"shower_inert_scale": 1.3}, mom=(0.0, 0.0, 600.0))
    np.testing.assert_array_equal(np.asarray(m0.edep), np.asarray(m1.edep))


def test_gradient_wrt_extras():
    keys = ["head_x0", "shower_tmax_shift", "lat_core", "lat_tail", "lat_pcore", "shower_inert_scale"]
    at = {"head_x0": 1.0, "shower_tmax_shift": 0.2, "lat_core": 0.8, "lat_tail": 1.1, "lat_pcore": 0.9,
          "shower_inert_scale": 1.1}
    det = ds.Detector(ds.Geometry(n_planes=48, n_cells=40))
    base = ds.default_params(det.geometry)

    def f(theta):
        p = ds.to_device(base, "float64")
        p.update({k: theta[i] for i, k in enumerate(keys)})
        ev = ds.make([11], [[5.0, -7.0, 40.0]], [[30.0, 60.0, 1200.0]], dtype=jnp.float64)
        q = reco.readout_charge(ds.expected(det, p, ev))
        return reco.visible_energy(det, p, q) + 1e3 * reco.theta2(det, q)

    testing.assert_gradient(f, jnp.array([at[k] for k in keys]), rel_step=1e-7, rtol=1e-4)


def test_materials_section_sets_derived_parameters():
    _, p0 = config.from_dict({})
    _, p1 = config.from_dict({"materials": {"inert": {"density": 1.5, "x0": 25.0}}})
    assert p1["rho_inert"] == 1.5 and p1["x0"] != p0["x0"]
    assert p1["rm"] == pytest.approx(21.2052 * p1["x0"] / p1["ec"])
    _, p2 = config.from_dict({"materials": {"inert": {"density": 1.5}}, "params": {"x0": 30.0}})
    assert p2["x0"] == 30.0                                            # params still override derived values
    with pytest.raises(KeyError):
        config.from_dict({"materials": {"glue": {"density": 1.0}}})
    with pytest.raises(KeyError):
        config.from_dict({"materials": {"inert": {"colour": 1.0}}})
