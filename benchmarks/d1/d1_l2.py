"""D1 level 2 (light response): fit the ND optics on the train chunks, then compare signal hits on the test chunks
(benchmarks/d1/criteria.json, L2).

    python benchmarks/d1/d1_l2.py OUT_DIR --description ND.json --stock CHUNK.npz ... --hepevt CHUNK.txt ...
        --chunk-ids ID ... [--batch 50] [--seed 7]

Signal hits: stock cell hits with BackTracker true energy > 0; detsim sampled hits (above threshold) with visible
energy > 0. Distance to the readout per hit, the same in both arms: the electron's straight axis (true vertex and
direction) at the hit's plane, d = L/2 - sign (w - w_centre). The fit (train) uses stock hits with more than 2 MeV of
true Birks energy, so the readout threshold does not bias it. Writes OUT_DIR/{reference,submission} for workflow
D1-L2, scores.json, record.json. Collaboration data: point OUT_DIR at a private area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_compare as D  # noqa: E402

DTYPES = ("float32", "float64")
FIT_BINS = np.arange(0.0, 400.0 + 1e-9, 10.0)
TEST_BINS = np.arange(0.0, 400.0 + 1e-9, 20.0)
FIT_MIN_MEV = 2.0


def load(a, which):
    """Selected events of the chunks in ``which`` (train/test): stock tables, matched particles, per-event truth."""
    import jax
    from novafastsim_detsim import particles
    ev_rows, parts, stock = [], [], []
    for cid, sf, hf in zip(a.chunk_ids, a.stock, a.hepevt):
        if (cid < 10) != (which == "train"):
            continue
        d = dict(np.load(sf))
        p = particles.from_hepevt(hf.read_text())
        hep_rows = d["ev_evt"].astype(int) - 1
        mom = np.asarray(p.momentum)[hep_rows, 0]
        if not np.allclose(np.sqrt(np.sum(mom ** 2, -1) + 0.511 ** 2) / 1000.0, d["ev_eE"], rtol=1e-4):
            raise SystemExit(f"{sf}: stock events do not match {hf}")
        s = D.selected(d)
        stock.append((d, np.flatnonzero(s)))
        parts.append(jax.tree_util.tree_map(lambda x: np.asarray(x)[hep_rows][s], p))
    return stock, jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *parts)


def axis_w(vtx, mom, z, view):
    """Long-axis coordinate of the electron's straight axis at plane z (view 0: w = y, view 1: w = x)."""
    u = mom / np.linalg.norm(mom)
    t = np.where(np.abs(u[2]) > 0.05, (z - vtx[2]) / np.where(np.abs(u[2]) > 0.05, u[2], 1.0), 0.0)
    x, y = vtx[0] + t * u[0], vtx[1] + t * u[1]
    return np.where(view == 0, y, x)


def distance(geo, plane, view, w):
    half = 0.5 * geo["length"]
    sign = np.where(view == 0, geo["readout"][0], geo["readout"][1])
    return np.clip(half - sign * (w - geo["wc"][plane]), 0.0, 2.0 * half)


def stock_hits(stock, geo, group=32):
    """Per signal hit: event index, PE, true Birks MeV, distance, ADC, and the FLS Birks MeV of the other cells on the
    hit's front-end board (``group`` cells of its plane); per event: n signal hits, signal PE, E, noise."""
    hits, evs = [], []
    k = 0
    for d, rows in stock:
        for row in rows:
            s = (d["hit_row"] == row) & (d["hit_plane"] < 192)
            sig = s & (d["hit_true_e"] > 0)
            noise = s & (d["hit_true_e"] <= 0)
            pl, vw = d["hit_plane"][sig], d["hit_view"][sig].astype(int)
            dm = (d["dep_row"] == row) & (d["dep_plane"] < 192)
            bkey = d["dep_plane"][dm].astype(np.int64) * 1000 + d["dep_cell"][dm] // group
            bsum = {}
            for kk, ee in zip(bkey.tolist(), (d["dep_edep_birks"][dm] * 1000.0).tolist()):
                bsum[kk] = bsum.get(kk, 0.0) + ee
            own = {}
            for p_, c_, ee in zip(d["dep_plane"][dm].tolist(), d["dep_cell"][dm].tolist(),
                                  (d["dep_edep_birks"][dm] * 1000.0).tolist()):
                own[(p_, c_)] = own.get((p_, c_), 0.0) + ee
            others = np.array([bsum.get(int(p_) * 1000 + int(c_) // group, 0.0) - own.get((int(p_), int(c_)), 0.0)
                               for p_, c_ in zip(pl, d["hit_cell"][sig])])
            vtx = np.array([d["ev_vx"][row], d["ev_vy"][row], d["ev_vz"][row]])
            mom = np.array([d["ev_epx"][row], d["ev_epy"][row], d["ev_epz"][row]])
            w = axis_w(vtx, mom, geo["z"][pl], vw)
            hits.append(np.c_[np.full(sig.sum(), k), d["hit_pe"][sig], d["hit_true_e_birks"][sig] * 1000.0,
                              distance(geo, pl, vw, w), d["hit_adc"][sig], others])
            evs.append((sig.sum(), d["hit_pe"][sig].sum(), d["ev_eE"][row], noise.sum(), d["hit_pe"][noise].sum()))
            k += 1
    return np.concatenate(hits), np.array(evs)


def ratio_profile(h, bins, n_events, min_mev=0.0):
    """Sum PE / sum MeV per distance bin, error by the delta method over events."""
    s = h[:, 2] > min_mev
    ev, pe, mev, dist = h[s, 0].astype(int), h[s, 1], h[s, 2], h[s, 3]
    b = np.clip(np.digitize(dist, bins) - 1, 0, len(bins) - 2)
    nb = len(bins) - 1
    A = np.zeros((n_events, nb)); B = np.zeros((n_events, nb))
    np.add.at(A, (ev, b), pe); np.add.at(B, (ev, b), mev)
    sa, sb = A.sum(0), B.sum(0)
    R = np.where(sb > 0, sa / np.where(sb > 0, sb, 1.0), 0.0)
    var = np.where(sb > 0, ((A - R * B) ** 2).sum(0) / np.where(sb > 0, sb, 1.0) ** 2, 0.0)
    return R, np.sqrt(var), sb


def fit_optics(det, params, R, E, centres):
    """yield_per_mev and the attenuation (frac_short, atten_short, atten_long) to the stock PE / MeV profile."""
    import jax.numpy as jnp
    from scintfibrefastsim.response import fibre_transport
    from novafastsim_detsim import config
    cell = det.geometry.cell()
    opt0, qe = config.optics(params), float(params["apd.qe"])
    use = E > 0
    wgt = np.where(use, 1.0 / np.where(use, E, 1.0) ** 2, 0.0)

    def model(q):
        f, a_s, a_l = 1 / (1 + np.exp(-q[0])), np.exp(q[1]), np.exp(q[2])
        o = opt0._replace(frac_short=f, atten_short=a_s, atten_long=a_l)
        c = np.asarray(fibre_transport(jnp.asarray(centres * 10.0), cell, o)[0], np.float64)
        y = np.sum(c * R * wgt) / np.sum(c * c * wgt) / qe
        return y, c, (f, a_s, a_l)

    def resid(q):
        y, c, _ = model(q)
        return ((y * qe * c - R) / np.where(use, E, 1.0))[use]

    q = np.array([0.0, np.log(3000.0), np.log(9000.0)])
    lam = 1e-3
    for _ in range(300):
        r = resid(q)
        J = np.stack([(resid(q + h) - r) / 1e-6 for h in np.eye(3) * 1e-6], axis=1)
        qn = q + np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12), -J.T @ r)
        qn[1:] = np.clip(qn[1:], np.log(100.0), np.log(200000.0))
        if np.sum(resid(qn) ** 2) < np.sum(r ** 2):
            q, lam = qn, lam / 3
        else:
            lam *= 5
    y, _, (f, a_s, a_l) = model(q)
    r = resid(q)
    return {"optics.yield_per_mev": float(y), "optics.frac_short": float(f), "optics.atten_short": float(a_s),
            "optics.atten_long": float(a_l), "chi2": float(np.sum(r ** 2)), "bins": int(use.sum())}


def detsim_hits(det, params, parts, geo, dtype, batch, seed):
    import jax
    import jax.numpy as jnp
    from novafastsim_detsim import config, sim
    p = config.to_device(params, getattr(jnp, dtype))
    key = jax.random.PRNGKey(seed)
    n = np.asarray(parts.pdg).shape[0]
    hits, evs = [], []
    P, C = det.geometry.n_planes, det.geometry.n_cells
    planes = np.repeat(np.arange(P), C).reshape(P, C)
    for i in range(0, n, batch):
        sl = np.arange(i, min(i + batch, n))
        sub = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[sl], getattr(jnp, dtype))
                                     if np.issubdtype(np.asarray(a).dtype, np.floating) else jnp.asarray(np.asarray(a)[sl]), parts)
        r, h = sim.sampled_batch(det, p, sub, jax.random.fold_in(key, i))
        evis, pe, hit = np.asarray(r.evis, np.float64), np.asarray(h.pe, np.float64), np.asarray(h.hit)
        vtx, mom = np.asarray(sub.vertex, np.float64)[:, 0], np.asarray(sub.momentum, np.float64)[:, 0]
        for k, ev in enumerate(sl):
            sig = hit[k] & (evis[k] > 0)
            pl = planes[sig]
            vw = geo["view"][pl]
            w = axis_w(vtx[k], mom[k], geo["z"][pl], vw)
            hits.append(np.c_[np.full(sig.sum(), ev), pe[k][sig], evis[k][sig], distance(geo, pl, vw, w)])
            e_true = float(np.sqrt(np.sum(mom[k] ** 2) + 0.511 ** 2)) / 1000.0
            evs.append((sig.sum(), pe[k][sig].sum(), e_true, 0, 0.0))
    return np.concatenate(hits), np.array(evs)


def l2_hists(hits, evs, pe_frac_range):
    n = len(evs)
    R, err, _ = ratio_profile(hits, TEST_BINS, n)
    pe_edges = np.geomspace(5.0, 5000.0, 61)
    c_pe, _ = np.histogram(np.clip(hits[:, 1], 5.0, 4999.0), pe_edges)
    return {"hists": {
        "nsig_hits": D.counts(evs[:, 0], 0.0, 500.0, 50),
        "pe_per_mev_vs_dist": {"edges": TEST_BINS.tolist(), "values": R.tolist(), "errors": err.tolist()},
        "hit_pe": {"edges": pe_edges.tolist(), "counts": c_pe.tolist()},
        "sig_pe_frac": D.counts(evs[:, 1] / evs[:, 2], *pe_frac_range, 50)}}


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from hepsimbench.score import score
    from novafastsim_detsim import config
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--params", type=Path, nargs="*", default=[],
                    help="fit outputs (d1_fit_deposits.py, d1_fit_stochastic.py) applied before the optics fit, in order")
    ap.add_argument("--fit-max-adc", type=float,
                    help="train optics fit: only stock hits below this ADC (keeps saturated hits out of the yield)")
    ap.add_argument("--fit-quiet-mev", type=float,
                    help="train optics fit: only stock hits whose board's other cells hold less than this Birks MeV")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="post-hoc parameter overrides applied after the fit (diagnostics only; recorded)")
    a = ap.parse_args(argv)
    det, params = config.load_description(a.description)
    det, deposit_fit = D.apply_fit(det, params, a.params)
    g = det.geometry
    lay = layout(det, jax.numpy.float64)
    geo = {"z": np.asarray(lay.z_lo) + 0.5 * g.cell_depth, "view": np.asarray(lay.view), "wc": np.asarray(lay.w_centre),
           "length": g.cell_length, "readout": g.readout_end}

    st_train, _ = load(a, "train")
    grp = det.settings.apd_group or 32
    h_tr, e_tr = stock_hits(st_train, geo, grp)
    h_fit = h_tr if a.fit_max_adc is None else h_tr[h_tr[:, 4] < a.fit_max_adc]
    if a.fit_quiet_mev is not None:
        h_fit = h_fit[h_fit[:, 5] < a.fit_quiet_mev]
    R, E, _ = ratio_profile(h_fit, FIT_BINS, len(e_tr), FIT_MIN_MEV)
    centres = 0.5 * (FIT_BINS[1:] + FIT_BINS[:-1])
    fit = fit_optics(det, params, R, E, centres)
    pe_frac = e_tr[:, 1] / e_tr[:, 2]
    rng = (0.0, float(np.quantile(pe_frac, 0.995)) * 1.1)
    p_fit = dict(params)
    p_fit.update({k: v for k, v in fit.items() if k.startswith("optics.")})
    post_hoc = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.set}
    p_fit.update(post_hoc)

    st_test, parts = load(a, "test")
    h_te, e_te = stock_hits(st_test, geo, grp)
    ref = a.out / "reference" / "D1-L2"
    (ref / "stock").mkdir(parents=True, exist_ok=True)
    (ref / "stock" / "hists.json").write_text(json.dumps(l2_hists(h_te, e_te, rng)))
    (ref / "reference.json").write_text(json.dumps({"stock_rate_per_slot": None, "mode": "same_events", "split": "test",
                                                    "boundary": "HepEVT electron -> signal cell hits (PE)"}))
    for dt in DTYPES:
        h_d, e_d = detsim_hits(det, p_fit, parts, geo, dt, a.batch, a.seed)
        sub = a.out / "submission" / "D1-L2" / dt
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "hists.json").write_text(json.dumps(l2_hists(h_d, e_d, rng)))
    res = score(a.out / "submission", a.out / "reference")
    (a.out / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    rec = {"criteria_sha256": hashlib.sha256((HERE / "criteria.json").read_bytes()).hexdigest(),
           "train_events": int(len(e_tr)), "test_events": int(len(e_te)), "fit": fit,
           "fit_profile": {"centres": centres.tolist(), "stock": R.tolist(), "errors": E.tolist()},
           "noise_test": {"hits_per_event": float(e_te[:, 3].mean()), "pe_per_event": float(e_te[:, 4].mean())},
           "sig_pe_frac_range": rng, "post_hoc_overrides": post_hoc, "fits": deposit_fit,
           "n_stoch": det.settings.n_stoch, "fit_max_adc": a.fit_max_adc, "fit_quiet_mev": a.fit_quiet_mev,
           "L2": {dt: {h["hist"]: {k: h[k] for k in ("chi2ndf", "ndf", "p", "flag")} for h in r["hists"]}
                  for dt, r in res["workflows"]["D1-L2"].items()}}
    (a.out / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("train_events", "test_events", "fit", "noise_test", "L2")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
