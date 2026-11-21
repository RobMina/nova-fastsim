"""Deposit models, response, sampling and I/O of the on-device detector simulation."""

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
from novafastsim_detsim import config, deposits, ranges, reco, sim  # noqa: E402
sff = pytest.importorskip("scintfibrefastsim")
from scintfibrefastsim import response as sff_response  # noqa: E402

BIG = ds.Detector(ds.Geometry(n_planes=128, n_cells=96))
SMALL = ds.Detector(ds.Geometry(n_planes=24, n_cells=32))


def P(det, dtype="float64", over=None):
    p = ds.to_device(ds.default_params(det.geometry), dtype)
    p.update({k: jnp.asarray(v, p["x0"].dtype) for k, v in (over or {}).items()})
    return p


def one(pdg, vtx, mom, dtype=jnp.float64):
    return ds.make([pdg], [vtx], [mom], dtype=dtype)


def test_contained_muon_closes_energy_and_range():
    det, params = BIG, P(BIG)
    T = 500.0
    m = 105.6583755
    p = np.sqrt((T + m) ** 2 - m * m)
    r = ds.expected(det, params, one(13, [1.7, 2.3, 50.0], [0.0, 0.0, p]))
    d = {k: float(v) for k, v in r.diag.items()}
    assert d["e_total"] == pytest.approx(T, rel=1e-9)
    assert d["e_scint"] + d["e_inert"] == pytest.approx(T, rel=1e-9)
    assert d["overflow"] == 0
    Tt, Rt = ranges.range_table(m)
    reach = np.interp(T, Tt, Rt) / float(params["rho_stop"])
    z_last = BIG.geometry.z_front + (np.flatnonzero(np.asarray(r.edep).sum(1))[-1] + 0.5) * BIG.geometry.plane_pitch
    assert abs(50.0 + reach - z_last) < BIG.geometry.plane_pitch


def test_muon_range_table_is_plausible():
    Tt, Rt = ranges.range_table(105.6583755)
    assert 400 < np.interp(1000.0, Tt, Rt) < 500   # g/cm^2 at 1 GeV in CH2


def _mean_depth(det, params, pdg, E):
    r = ds.expected(det, params, one(pdg, [0.0, 0.0, 30.0], [0.0, 0.0, E]))
    w = np.asarray(r.edep).sum(1)
    z = det.geometry.z_front + (np.arange(det.geometry.n_planes) + 0.5) * det.geometry.plane_pitch
    return (w * z).sum() / w.sum() - 30.0, r


def test_shower_closes_energy_and_follows_the_gamma_profile():
    det, params = BIG, P(BIG)
    E = 2000.0
    depth, r = _mean_depth(det, params, 11, E)
    d = {k: float(v) for k, v in r.diag.items()}
    T = np.hypot(E, 0.51099895) - 0.51099895
    assert d["e_total"] == pytest.approx(T, rel=1e-9)
    assert d["e_outside"] < 1e-3 * E and d["overflow"] == 0
    a, _ = deposits.longitudinal(jnp.asarray(E), params["ec"], params["shower_b"], False)
    assert depth == pytest.approx(float(a / params["shower_b"] * params["x0"]), rel=0.05)


def test_photon_shower_starts_one_radiation_length_later():
    det, params = BIG, P(BIG)
    de, _ = _mean_depth(det, params, 11, 1500.0)
    dg, _ = _mean_depth(det, params, 22, 1500.0)
    assert dg - de == pytest.approx(float(params["x0"]), rel=0.1)


def test_birks_off_gives_visible_equal_deposited():
    params = P(SMALL, over={"optics.birks_kb": 0.0, "optics.birks_kc": 0.0})
    r = ds.expected(SMALL, params, one(13, [3.0, -4.0, 5.0], [50.0, 20.0, 400.0]))
    np.testing.assert_allclose(np.asarray(r.evis), np.asarray(r.edep), rtol=1e-12)


def test_cell_response_is_scint_fibre_fastsim():
    # an axial muon through one cell column: per-cell PE = sff light(edep, dE/dx, distance) x APD quantum efficiency
    det, params = SMALL, P(SMALL)
    g = det.geometry
    x = -g.half_width + 4.5 * g.cell_pitch
    y = -g.half_width + 20.5 * g.cell_pitch
    r = ds.expected(det, params, one(13, [x, y, -1.0], [0.0, 0.0, 5000.0]))
    p = 1                                                        # a y-view plane: cell 20, read out at +x
    edep = float(r.edep[p, 20])
    dedx = edep / g.cell_depth / 10.0                            # MeV / mm
    cell, opt = g.cell(), config.optics(params)
    photons, _ = sff_response.light(np.array([edep]), np.array([dedx]), np.array([(g.half_width - x) * 10.0]),
                                    cell, sff_response.CellOptics(*(float(v) for v in opt)))
    assert float(r.pe[p, 20]) == pytest.approx(photons[0] * float(params["apd.qe"]), rel=1e-4)  # dE/dx varies by segment
    assert g.cell().width == sff.geometry.nova_cell_tdr().width


def test_optics_table_matches_analytic():
    det = SMALL
    params = P(det)
    ev = one(11, [2.0, 1.0, 20.0], [40.0, -30.0, 800.0])
    tab = sff_response.tabulate(det.geometry.cell(), sff_response.CellOptics(*(float(v) for v in config.optics(params))),
                                n=1025)
    r_a = ds.expected(det, params, ev)
    r_t = ds.expected(det, dict(params, optics_table=config.to_device({"t": tab}, "float64")["t"]), ev)
    np.testing.assert_allclose(np.asarray(r_t.pe), np.asarray(r_a.pe), rtol=1e-4, atol=1e-6)


def test_sampled_readout_is_centred_on_the_expected_response():
    params = P(SMALL)
    ev = one(11, [2.0, 1.0, 20.0], [40.0, -30.0, 800.0])
    r = ds.expected(SMALL, params, ev)
    keys = jax.random.split(jax.random.PRNGKey(7), 400)
    pe = jax.vmap(lambda k: ds.sampled(SMALL, params, ev, k)[1].pe)(keys)
    mean, mu = np.asarray(pe).mean(0), np.asarray(r.pe)
    F = float(params["apd.excess_noise"])
    lit = mu > 5
    z = (mean[lit] - mu[lit]) / np.sqrt(F * mu[lit] / len(keys))
    assert abs(z.mean()) < 0.3 and np.abs(z).max() < 5.5


def test_expected_hit_probability_matches_sampled_hit_rate():
    params = P(SMALL)
    ev = one(13, [2.0, 1.0, 1.0], [600.0, 0.0, 300.0])
    r = ds.expected(SMALL, params, ev)
    keys = jax.random.split(jax.random.PRNGKey(3), 300)
    hit = jax.vmap(lambda k: ds.sampled(SMALL, params, ev, k)[1].hit)(keys)
    assert float(jnp.mean(hit.sum((1, 2)))) == pytest.approx(float(r.p_hit.sum()), rel=0.05)


def test_float32_matches_float64():
    ev64 = one(11, [2.0, 1.0, 20.0], [40.0, -30.0, 800.0])
    ev32 = one(11, [2.0, 1.0, 20.0], [40.0, -30.0, 800.0], dtype=jnp.float32)
    r64 = ds.expected(SMALL, P(SMALL), ev64)
    r32 = ds.expected(SMALL, P(SMALL, "float32"), ev32)
    assert r32.pe.dtype == jnp.float32
    assert float(r32.pe.sum()) == pytest.approx(float(r64.pe.sum()), rel=1e-4)
    np.testing.assert_allclose(np.asarray(r32.pe), np.asarray(r64.pe), rtol=1e-3, atol=1e-3)


def test_batch_matches_single_events():
    params = P(SMALL)
    evs = ds.make([[11, 13], [22, 0]], [[[1, 2, 10], [0, 0, 5]], [[-3, 4, 30], [0, 0, 0]]],
                  [[[0, 0, 500], [100, 0, 400]], [[10, 10, 700], [0, 0, 0]]],
                  mask=[[True, True], [True, False]], dtype=jnp.float64)
    rb = ds.expected_batch(SMALL, params, evs)
    for e in range(2):
        ev = jax.tree_util.tree_map(lambda a: a[e], evs)
        np.testing.assert_allclose(np.asarray(rb.pe[e]), np.asarray(ds.expected(SMALL, params, ev).pe), rtol=1e-12)


def test_non_interacting_particles_deposit_nothing():
    r = ds.expected(SMALL, P(SMALL), ds.make([18, 14, 2112], [[0, 0, 10]] * 3, [[0, 0, 500]] * 3,
                                             dtype=jnp.float64))
    assert float(r.diag["e_total"]) == 0.0 and float(r.pe.sum()) == 0.0


def test_from_hepevt_units_and_status():
    text = ("0 2\n"
            "1 11 0 -1 -1 -1 0.01 0.02 1.5 1.5002 0.000511 12.0 -3.0 150.0 7.5\n"
            "2 18 0 -1 -1 -1 0 0 1 1.2 0.5 12.0 -3.0 150.0 7.5\n"
            "1 1\n"
            "1 13 0 -1 -1 -1 0 0 0.5 0.51 0.1057 0 0 10 0\n")
    p = ds.from_hepevt(text)
    assert p.pdg.shape == (2, 1)
    assert int(p.pdg[0, 0]) == 11 and int(p.pdg[1, 0]) == 13
    np.testing.assert_allclose(np.asarray(p.momentum[0, 0]), [10.0, 20.0, 1500.0])
    np.testing.assert_allclose(np.asarray(p.vertex[0, 0]), [12.0, -3.0, 150.0])
    assert float(p.time[0, 0]) == 7.5


def test_from_bdnmc_converts_frame_and_units():
    e4 = jnp.array([[1.5, 0.0, 0.1, 1.49]])
    v4 = jnp.array([[1.0, 2.0, 103.0, 3.4e-7]])
    p = ds.from_bdnmc(e4, v4, origin_m=(1.0, 2.0, 100.0))
    np.testing.assert_allclose(np.asarray(p.vertex[0]), [0.0, 0.0, 300.0])
    np.testing.assert_allclose(np.asarray(p.momentum[0]), [0.0, 100.0, 1490.0])
    assert float(p.time[0]) == pytest.approx(340.0)


def test_config_round_trip_and_fail_closed(tmp_path):
    det, params = ds.from_dict({"geometry": {"n_planes": 10, "readout_end": [1, -1]}, "params": {"optics.yield_per_mev": 7.0}})
    assert det.geometry.n_planes == 10 and det.geometry.readout_end == (1, -1) and params["optics.yield_per_mev"] == 7.0
    f = tmp_path / "det.json"
    f.write_text(json.dumps(ds.config.to_dict(det, params)))
    det2, params2 = ds.load(f)
    assert det2 == det and params2["optics.yield_per_mev"] == 7.0
    for bad in ({"geometry": {"n_plane": 3}}, {"params": {"pe_per_MeV": 1}}, {"detector": {}}):
        with pytest.raises(KeyError):
            ds.from_dict(bad)


def test_reco_angle_and_attenuation_corrected_energy():
    det = BIG
    params = P(det)

    def run(E, th):
        ev = one(11, [3.0, -2.0, 40.0], [E * np.sin(th), 0.0, E * np.cos(th)])
        r = ds.expected(det, params, ev)
        q = reco.readout_charge(r)
        return (float(r.evis.sum()), float(reco.visible_energy(det, params, r.pe)),
                float(reco.visible_energy(det, params, q)), float(reco.theta2(det, q)))

    out = [run(E, 0.15) for E in (500.0, 1000.0, 2500.0)]
    for evis, all_charge, read_out, t2 in out:
        assert all_charge == pytest.approx(evis, rel=0.01)    # attenuation correction closes
        assert 0.3 * evis < read_out < evis                  # threshold losses
        assert t2 == pytest.approx(0.15 ** 2, rel=0.25)
    fr = [o[2] / o[0] for o in out]
    assert fr[0] < fr[1] < fr[2]                             # losses shrink with energy
