"""The stochastic deposit tier: energy closure per draw, mean = expected, clumpier showers, gradients at fixed keys."""

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
testing = pytest.importorskip("hepsimlib.testing")
from scintfibrefastsim.backends import nova_apd  # noqa: E402

from novafastsim_detsim import config, reco, sim  # noqa: E402

GEOM = ds.Geometry(n_planes=128, n_cells=96)
DET0 = ds.Detector(GEOM)
DET = ds.Detector(GEOM, replace(DET0.settings, n_stoch=192))
BASE = ds.default_params(GEOM)
T_KIN = np.hypot(2000.0, 0.51099895) - 0.51099895


def params(over=None, dtype="float64"):
    p = ds.to_device(BASE, dtype)
    p.update({k: jnp.asarray(v, p["x0"].dtype) for k, v in (over or {}).items()})
    return p


def electron(dtype=jnp.float64, mom=(0.0, 0.0, 2000.0)):
    return ds.make([11], [[1.0, 0.7, 30.0]], [list(mom)], dtype=dtype)


def test_off_by_default_and_sampled_unchanged():
    assert DET0.settings.n_stoch == 0
    p, ev, key = params(), electron(), jax.random.PRNGKey(3)
    r, h = sim.sampled(DET0, p, ev, key)
    e = sim.expected(DET0, p, ev)
    np.testing.assert_allclose(np.asarray(r.pe), np.asarray(e.pe), rtol=1e-12, atol=1e-12)
    ref = nova_apd.sample(key, e.pe, e.time, config.apd(p), DET0.settings.readout())
    np.testing.assert_allclose(np.asarray(h.pe), np.asarray(ref[0]), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        sim.sampled_deposits(DET0, p, ev, key)


@pytest.mark.parametrize("frac", [0.3, 1.0])
def test_every_draw_closes_energy(frac):
    p = params({"stoch_frac": frac, "stoch_tmax_sigma": 0.5})
    for s in range(3):
        r = sim.sampled_deposits(DET, p, electron(), jax.random.PRNGKey(s))
        d = {k: float(v) for k, v in r.diag.items()}
        assert d["e_total"] == pytest.approx(T_KIN, rel=1e-9)
        assert d["e_scint"] + d["e_inert"] + d["e_outside"] == pytest.approx(T_KIN, rel=1e-6)
        assert d["overflow"] == 0


def test_mean_of_draws_follows_the_expected_shower():
    p = params({"stoch_frac": 1.0, "stoch_theta": 0.3})
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    evs = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (64,) + a.shape), electron())
    r = jax.vmap(lambda e, k: sim.sampled_deposits(DET, p, e, k))(evs, keys)
    mean_planes = np.asarray(r.edep).sum(-1).mean(0)
    exp_planes = np.asarray(sim.expected(DET, p, electron()).edep).sum(-1)
    assert mean_planes.sum() == pytest.approx(exp_planes.sum(), rel=0.03)
    z = np.arange(len(exp_planes))
    depth = lambda w: (w * z).sum() / w.sum()
    assert depth(mean_planes) == pytest.approx(depth(exp_planes), abs=0.5)    # planes
    cum_m, cum_e = np.cumsum(mean_planes) / mean_planes.sum(), np.cumsum(exp_planes) / exp_planes.sum()
    assert np.max(np.abs(cum_m - cum_e)) < 0.05


def test_draws_are_clumpier_than_the_mean_shower():
    p = params({"stoch_frac": 1.0})
    e = np.asarray(sim.expected(DET, p, electron()).edep)
    s = np.asarray(sim.sampled_deposits(DET, p, electron(), jax.random.PRNGKey(1)).edep)
    assert (s > 0.5).sum() < 0.6 * (e > 0.5).sum()
    assert np.median(s[s > 0.1]) > 4.0 * np.median(e[e > 0.1])       # track crossings of a few MeV, not a haze


def test_tmax_sigma_spreads_shower_depths():
    def depths(sig):
        p = params({"stoch_frac": 0.5, "stoch_tmax_sigma": sig})
        out = []
        for s in range(12):
            w = np.asarray(sim.sampled_deposits(DET, p, electron(), jax.random.PRNGKey(s)).edep).sum(-1)
            out.append((w * np.arange(len(w))).sum() / w.sum())
        return np.std(out)
    assert depths(1.5) > 2.0 * depths(0.0)


def test_gradient_wrt_stochastic_parameters_at_fixed_key():
    keys = ["stoch_frac", "stoch_len", "stoch_dedx", "stoch_theta", "stoch_tmax_sigma", "shower_b"]
    at = {"stoch_frac": 0.6, "stoch_len": 7.0, "stoch_dedx": 2.1, "stoch_theta": 0.4, "stoch_tmax_sigma": 0.3,
          "shower_b": 0.55}
    g = ds.Geometry(n_planes=48, n_cells=40)
    det = ds.Detector(g, replace(ds.Detector(g).settings, n_stoch=48))
    base = ds.default_params(g)
    key = jax.random.PRNGKey(5)

    def f(theta):
        p = ds.to_device(base, "float64")
        p.update({k: theta[i] for i, k in enumerate(keys)})
        ev = ds.make([11], [[5.0, -7.0, 40.0]], [[30.0, 60.0, 1200.0]], dtype=jnp.float64)
        q = reco.readout_charge(sim.sampled_deposits(det, p, ev, key))
        return reco.visible_energy(det, p, q)

    testing.assert_gradient(f, jnp.array([at[k] for k in keys]), rel_step=1e-7, rtol=1e-4)


def test_float32_draw_closes_energy():
    r = sim.sampled_deposits(DET, params({"stoch_frac": 0.7}, "float32"), electron(jnp.float32), jax.random.PRNGKey(2))
    d = {k: float(v) for k, v in r.diag.items()}
    assert r.edep.dtype == jnp.float32
    assert d["e_scint"] + d["e_inert"] + d["e_outside"] == pytest.approx(T_KIN, rel=1e-4)


def test_apd_groups_couple_cells_of_one_board():
    g = ds.Geometry(n_planes=4, n_cells=96)
    det = ds.Detector(g, replace(ds.Detector(g).settings, apd_group=32))
    grp = np.asarray(sim.apd_groups(det))
    assert grp.shape == (4, 96) and grp[0, 31] == grp[0, 0] and grp[0, 32] != grp[0, 31] and grp[1, 0] == 3
    assert sim.apd_groups(ds.Detector(g)) is None
    p0 = ds.to_device(ds.default_params(g), "float64")
    p1 = dict(p0, **{"apd.crosstalk": jnp.asarray(0.03, jnp.float64)})
    ev = ds.make([11], [[1.0, 0.7, 2.0]], [[0.0, 0.0, 1500.0]], dtype=jnp.float64)
    r0, r1 = sim.expected(det, p0, ev), sim.expected(det, p1, ev)
    assert float(r1.p_hit.sum()) < float(r0.p_hit.sum())
    np.testing.assert_allclose(np.asarray(r1.pe), np.asarray(r0.pe))       # light unchanged; the readout couples
