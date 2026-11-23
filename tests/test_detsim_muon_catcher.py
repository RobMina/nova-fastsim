"""Planes of fewer and shorter cells, region boxes and absorber boxes (a muon-catcher-like tail) on a synthetic
detector."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ds = pytest.importorskip("novafastsim_detsim")
testing = pytest.importorskip("hepsimlib.testing")
from novafastsim_detsim import config, geometry, reco  # noqa: E402

PITCH, NMAIN, CMAIN, CMC = 6.65, 10, 20, 12
STEEL = [-40.0, 40.0, -40.0, 10.0, 84.0, 94.0]


def description(with_steel=True):
    main = [{"z": 3.3 + PITCH * k, "view": k % 2, "u": [-37.05 + 3.9 * c for c in range(CMAIN)],
             "narrow": [False] * CMAIN, "long_centre": 0.0} for k in range(NMAIN)]
    mc = [{"z": 78.0 + 22.0 * j, "view": (NMAIN + j) % 2, "u": [-37.05 + 3.9 * c for c in range(CMC)],
           "narrow": [False] * CMC, "long_centre": -15.0, "length": 50.0} for j in range(2)]
    d = {"constants": {"cell_depth": 5.0, "cell_width_wide": 3.5, "cell_width_narrow": 3.4, "web": 0.4, "wall": 0.5,
                       "cell_length": 80.0},
         "planes": main + mc,
         "regions": [[-40.0, 40.0, -40.0, 40.0, 0.0, 66.5]] + [[-40.0, 40.0, -40.0, 10.0, p["z"] - 3.3, p["z"] + 3.3]
                                                             for p in mc],
         "materials": {"Steel": {"density_g_cm3": 7.84, "radlen_cm": 1.77}}}
    if with_steel:
        d["absorbers"] = [{"box": STEEL, "material": "Steel"}]
    return d


def setup(with_steel=True):
    det, params = config.from_dict(config.tables_from_description(description(with_steel)))
    return det, ds.to_device(params, "float64")


def muon(y=-5.0, T=1000.0):
    m = 105.6583755
    p = np.sqrt((T + m) ** 2 - m * m)
    return ds.make([13], [[1.0, y, 1.0]], [[0.0, 0.0, p]], dtype=jnp.float64)


def test_ragged_planes_are_padded_and_lengths_per_plane():
    det, _ = setup()
    g = det.geometry
    assert g.n_planes == NMAIN + 2 and g.n_cells == CMAIN
    assert all(w == 0.0 for w in g.cell_w[NMAIN][CMC:]) and g.cell_w[NMAIN][CMC - 1] == 3.5
    assert g.lengths()[NMAIN:] == (50.0, 50.0) and g.lengths()[0] == 80.0
    assert len(g.absorbers) == 1 and len(g.region_boxes()) == 3


def test_steel_takes_the_minimum_ionising_share():
    det, p = setup()
    r = ds.expected(det, p, muon())
    d = {k: float(v) for k, v in r.diag.items()}
    expect = float(p["rho_abs"] * p["stop_abs"]) * 10.0 * 2.2         # ~ dE/dx of a 1 GeV muon, scint-equivalent
    assert d["e_absorber"] == pytest.approx(expect, rel=0.25)
    assert d["e_scint"] + d["e_inert"] + d["e_absorber"] + d["e_outside"] == pytest.approx(d["e_total"], rel=1e-9)
    assert float(r.edep[NMAIN:].sum()) > 0                               # the catcher planes see the muon
    assert float(r.edep[NMAIN:, CMC:].sum()) == 0.0                      # padding cells never do


def test_steel_shortens_the_range():
    T = 220.0                                                             # stops between the two catcher planes
    with_s, without = (ds.expected(*setup(w), muon(T=T)) for w in (True, False))
    assert float(with_s.edep[NMAIN + 1].sum()) == 0.0 and float(without.edep[NMAIN + 1].sum()) > 0.0


def test_no_loss_in_the_gap_above_the_catcher():
    det, p = setup()
    low, high = ds.expected(det, p, muon(y=-5.0)), ds.expected(det, p, muon(y=30.0))
    assert float(high.diag["e_absorber"]) == 0.0 and float(high.edep[NMAIN:].sum()) == 0.0
    assert float(high.diag["e_total"]) < 0.7 * float(low.diag["e_total"])


def test_readout_distance_uses_the_plane_length():
    det, _ = setup()
    lay = geometry.layout(det, jnp.float64)
    np.testing.assert_allclose(np.asarray(lay.half_len)[[0, NMAIN]], [40.0, 25.0])
    d = geometry.readout_distance(det, jnp.array([0, 0]), jnp.array([-15.0, -40.0]), jnp.array([NMAIN, NMAIN]),
                                  jnp.float64)
    np.testing.assert_allclose(np.asarray(d), [25.0, 50.0])


def test_gradient_wrt_absorber_density():
    det, base = setup()

    def f(theta):
        p = dict(base, rho_abs=theta[0])
        q = reco.readout_charge(ds.expected(det, p, muon()))
        return reco.visible_energy(det, p, q)

    testing.assert_gradient(f, jnp.array([7.84]), rel_step=1e-7, rtol=1e-4)


def test_private_nd_description_with_muon_catcher():
    import os
    from pathlib import Path
    root = os.environ.get("NOVA_FASTSIM_INPUTS")
    path = Path(root or ".") / "nova" / "detector" / "nd_S26-07-01_mc.json"
    if not root or not path.exists():
        pytest.skip("private ND description with the muon catcher not available")
    det, params = config.load_description(path)
    g = det.geometry
    assert g.n_planes > 192 and len(g.absorbers) > 0 and len(g.region_boxes()) == g.n_planes - 192 + 1
    p = ds.to_device(params, "float64")
    m = 105.6583755
    mom = np.sqrt((2000.0 + m) ** 2 - m * m)
    ev = ds.make([13], [[0.0, -50.0, 1200.0]], [[0.0, 0.0, mom]], dtype=jnp.float64)
    r = ds.expected(det, p, ev)
    assert float(r.diag["e_absorber"]) > 0.0 and float(r.edep[192:].sum()) > 0.0
