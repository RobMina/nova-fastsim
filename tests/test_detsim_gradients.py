"""Autodiff through the full chain (particles -> cells -> readout -> reco) against central finite differences,
and a calibration-fit closure through hepsimlib.fit."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
testing = pytest.importorskip("hepsimlib.testing")
from novafastsim_detsim import calibrate, reco  # noqa: E402

DET = ds.Detector(ds.Geometry(n_planes=48, n_cells=40))
BASE = ds.default_params(DET.geometry)
VTX, MOM = [5.0, -7.0, 40.0], [30.0, 60.0, 1200.0]


def _observable(params, vtx, mom, pdg=11):
    ev = ds.make([pdg], [vtx], [mom], dtype=params["x0"].dtype)
    r = ds.expected(DET, params, ev)
    q = reco.readout_charge(r)
    return reco.visible_energy(DET, params, q) + 1e3 * reco.theta2(DET, q)


def test_gradient_wrt_kinematics():
    params = ds.to_device(BASE, "float64")
    f = lambda x: _observable(params, x[:3], x[3:])
    # shower nodes cross cell walls as the vertex moves (piecewise-linear response): small steps
    testing.assert_gradient(f, jnp.array(VTX + MOM), rel_step=1e-7, rtol=1e-5)


@pytest.mark.parametrize("pdg,mom", [(11, MOM), (13, [80.0, 40.0, 700.0])])
def test_gradient_wrt_response_parameters(pdg, mom):
    keys = ["optics.yield_per_mev", "optics.atten_long", "optics.atten_short", "optics.frac_short",
            "optics.birks_kb", "rm", "apd.tau_rise", "apd.threshold_pe", "apd.noise_pe", "optics.speed", "apd.qe"]

    def f(theta):
        p = ds.to_device(BASE, "float64")
        p.update({k: theta[i] for i, k in enumerate(keys)})
        return _observable(p, VTX, mom, pdg)

    testing.assert_gradient(f, jnp.array([BASE[k] for k in keys]))


def test_gradient_wrt_shower_shape_at_small_steps():
    # hard cell walls make the response piecewise linear in shower node positions: compare at small steps
    keys = ["x0", "shower_b", "ec"]

    def f(theta):
        p = ds.to_device(BASE, "float64")
        p.update({k: theta[i] for i, k in enumerate(keys)})
        return _observable(p, VTX, MOM)

    testing.assert_gradient(f, jnp.array([BASE[k] for k in keys]), rel_step=1e-7, rtol=1e-4)


def test_float32_gradient_is_usable():
    p32 = ds.to_device(BASE, "float32")
    k = "optics.yield_per_mev"
    g32 = jax.grad(lambda y: _observable({**p32, k: y}, VTX, MOM))(p32[k])
    p64 = ds.to_device(BASE, "float64")
    g64 = jax.grad(lambda y: _observable({**p64, k: y}, VTX, MOM))(p64[k])
    assert g32.dtype == jnp.float32
    assert float(g32) == pytest.approx(float(g64), rel=1e-3)


def test_calibration_fit_recovers_light_yield_and_attenuation():
    det = ds.Detector(ds.Geometry(n_planes=16, n_cells=32))
    truth = ds.to_device(ds.default_params(det.geometry), "float64")
    rng = np.random.default_rng(11)
    n = 24
    xy = rng.uniform(-55.0, 55.0, (n, 2))
    vtx = np.c_[xy, np.full(n, -1.0)][:, None, :]
    mom = np.tile([0.0, 0.0, 3000.0], (n, 1))[:, None, :]
    muons = ds.make(np.full((n, 1), 13), vtx, mom, dtype=jnp.float64)
    mu = np.asarray(ds.expected_batch(det, truth, muons).pe)
    observed = rng.poisson(mu)
    names = ("optics.yield_per_mev", "optics.atten_long")
    start = {**truth, names[0]: jnp.asarray(9.0), names[1]: jnp.asarray(5000.0)}
    res = calibrate.fit_poisson(det, start, muons, observed, names)
    assert res.result.converged
    pulls = (res.theta - np.array([float(truth[n]) for n in names])) / res.errors
    assert np.all(np.abs(pulls) < 4), (res.theta, res.errors)
