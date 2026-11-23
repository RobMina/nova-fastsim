"""D1 singles arm: detsim against the stock ND simulation on the same HepEVT electrons (benchmarks/d1/criteria.json).

    python benchmarks/d1/d1_compare.py OUT_DIR --description ND.json --stock CHUNK.npz ... --hepevt CHUNK.txt ...
        [--split train|test|all] [--batch 50] [--params FIT.json ...] [--sampled [--seed 11]]

``--stock`` files come from the private stock converter (one per chunk, rows in HepEVT event order) and pair with the
``--hepevt`` chunks in the order given. Writes OUT_DIR/reference/<W>/stock/hists.json (stock arm, reference.json),
OUT_DIR/submission/<W>/<dtype>/hists.json (detsim arm), OUT_DIR/scores.json and OUT_DIR/record.json. Stock data
and results are collaboration data: point OUT_DIR at a private area. Needs JAX and hep-simbench.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DTYPES = ("float32", "float64")
SEL = dict(xy=170.0, z_lo=20.0, z_hi=1250.0, e_lo=0.1, e_hi=10.0)
N_LONG, N_LAT = 120, 30


# ------------------------------------------------------------------------------------------- inputs
def apply_fit(det, params, paths):
    """Apply fit outputs (``d1_fit_deposits.py``, ``d1_fit_stochastic.py``) in order: each file's ``params`` dict
    updates ``params`` in place, its optional ``settings`` dict replaces Settings fields and its optional
    ``optics_table`` (distance mm, collection, delay ns) becomes ``params["optics_table"]``. Returns the new detector
    and the fits' records for the scoring record."""
    from dataclasses import fields, replace
    recs = []
    for path in paths or []:
        raw = path.read_bytes()
        fit = json.loads(raw)
        unknown = sorted(set(fit["params"]) - set(params))
        if unknown:
            raise SystemExit(f"{path}: unknown parameters {unknown}")
        params.update(fit["params"])
        sett = fit.get("settings", {})
        bad = sorted(set(sett) - {f.name for f in fields(det.settings)})
        if bad:
            raise SystemExit(f"{path}: unknown settings {bad}")
        det = replace(det, settings=replace(det.settings, **sett))
        if "optics_table" in fit:
            from scintfibrefastsim.response import CellTable
            t = fit["optics_table"]
            params["optics_table"] = CellTable(np.asarray(t["distance"], float), np.asarray(t["collection"], float),
                                               np.asarray(t["delay"], float))
        recs.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                     "criteria_sha256": fit.get("criteria_sha256"), "params": fit["params"], "settings": sett})
    return det, recs


def selected(ev):
    return ((np.abs(ev["ev_vx"]) < SEL["xy"]) & (np.abs(ev["ev_vy"]) < SEL["xy"]) & (ev["ev_vz"] > SEL["z_lo"])
            & (ev["ev_vz"] < SEL["z_hi"]) & (ev["ev_eE"] > SEL["e_lo"]) & (ev["ev_eE"] < SEL["e_hi"]))


# ------------------------------------------------------------------------------------------- observables

def plane_of(z, plane_edges):
    return np.clip(np.searchsorted(plane_edges, z, side="right") - 1, 0, len(plane_edges) - 2)


def l1_event(edep_pc, evis_pc, e_true_mev, vtx_plane, view, P=192):
    """Level-1 observables of one event from per-cell deposits [P, C] (MeV)."""
    tot = edep_pc.sum()
    out = {"escint_frac": tot / e_true_mev, "birks_ratio": evis_pc.sum() / tot if tot > 0 else np.nan,
           "ncells": int((edep_pc > 0.5).sum())}
    per_plane = edep_pc.sum(axis=1)
    off = np.arange(P) - vtx_plane
    lp = np.zeros(N_LONG)
    m = (off >= 0) & (off < N_LONG)
    np.add.at(lp, off[m], per_plane[m])
    out["long_profile"] = lp / e_true_mev
    lat = np.zeros(N_LAT)
    C = edep_pc.shape[1]
    for v in (0, 1):
        sel = view == v
        e = edep_pc[sel]
        if e.sum() <= 0:
            continue
        cells = np.broadcast_to(np.arange(C), e.shape)
        mean = np.average(cells, weights=e)
        dist = np.minimum(np.rint(np.abs(cells - mean)).astype(int), N_LAT - 1)
        np.add.at(lat, dist.ravel(), e.ravel())
    out["lat_profile"] = lat / e_true_mev
    return out


def stock_cells(d, row, P=192, C=96):
    s = (d["dep_row"] == row) & (d["dep_plane"] < P)
    edep, evis = np.zeros((P, C)), np.zeros((P, C))
    np.add.at(edep, (d["dep_plane"][s], d["dep_cell"][s]), d["dep_edep"][s] * 1000.0)
    np.add.at(evis, (d["dep_plane"][s], d["dep_cell"][s]), d["dep_edep_birks"][s] * 1000.0)
    return edep, evis


# ------------------------------------------------------------------------------------------- histograms

H1 = {"escint_frac": (0.0, 1.0, 40), "birks_ratio": (0.9, 1.0, 40), "ncells": (0.0, 500.0, 50)}


def counts(vals, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    c, _ = np.histogram(np.clip(np.asarray(vals, float), lo, hi - 1e-9 * (hi - lo)), edges)
    return {"edges": edges.tolist(), "counts": c.tolist()}


def profile(rows, n):
    a = np.asarray(rows)
    return {"edges": np.arange(n + 1, dtype=float).tolist(), "values": a.mean(0).tolist(),
            "errors": (a.std(0, ddof=1) / np.sqrt(len(a))).tolist()}


def hists_l1(obs):
    h = {k: counts([o[k] for o in obs], *H1[k]) for k in H1}
    h["long_profile"] = profile([o["long_profile"] for o in obs], N_LONG)
    h["lat_profile"] = profile([o["lat_profile"] for o in obs], N_LAT)
    return {"hists": h}


# ------------------------------------------------------------------------------------------- detsim arm

def detsim_l1(det, params, parts, rows, dtype, batch, vtx_planes, view, seed=None):
    import jax
    import jax.numpy as jnp
    from novafastsim_detsim import config, sim
    p = config.to_device(params, getattr(jnp, dtype))
    out = []
    idx = np.asarray(rows)
    for i in range(0, len(idx), batch):
        sl = idx[i:i + batch]
        sub = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[sl], getattr(jnp, dtype))
                                     if np.issubdtype(np.asarray(a).dtype, np.floating) else jnp.asarray(np.asarray(a)[sl]), parts)
        if seed is None:
            r = sim.expected_batch(det, p, sub)
        else:
            r = sim.sampled_deposits_batch(det, p, sub, jax.random.fold_in(jax.random.PRNGKey(seed), i))
        edep, evis = np.asarray(r.edep, np.float64), np.asarray(r.evis, np.float64)
        e_true = np.asarray(sub.momentum, np.float64)
        for k, row in enumerate(sl):
            et = float(np.sqrt(np.sum(e_true[k, 0] ** 2) + 0.511 ** 2))
            out.append(l1_event(edep[k], evis[k], et, vtx_planes[row], view))
    return out


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from hepsimbench.score import score
    from novafastsim_detsim import config, particles
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", help="chunk index of each pair (default 0, 1, ...)")
    ap.add_argument("--split", choices=("train", "test", "all"), default="test")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--params", type=Path, nargs="*", default=[],
                    help="fit outputs (d1_fit_deposits.py, d1_fit_stochastic.py) applied to the description in order")
    ap.add_argument("--sampled", action="store_true",
                    help="one draw of the stochastic deposit tier per event instead of the expected deposits")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    if len(a.stock) != len(a.hepevt):
        raise SystemExit("--stock and --hepevt need the same number of files")
    ids = a.chunk_ids or list(range(len(a.stock)))
    use = {"train": lambda i: i < 10, "test": lambda i: i >= 10, "all": lambda i: True}[a.split]

    det, params = config.load_description(a.description)
    det, deposit_fit = apply_fit(det, params, a.params)
    if a.sampled and det.settings.n_stoch <= 0:
        raise SystemExit("--sampled needs a fit file that sets n_stoch > 0")
    lay = layout(det)
    pe_edges = np.asarray(lay.plane_edges, np.float64)
    view = np.asarray(lay.view)
    stock_obs, sel_parts, vtx_planes, n_ev = [], [], [], 0
    for cid, sf, hf in zip(ids, a.stock, a.hepevt):
        if not use(cid):
            continue
        d = dict(np.load(sf))
        p = particles.from_hepevt(hf.read_text())
        # stock event k is HepEVT event k (1-based); the generation filter drops events without hits
        hep_rows = d["ev_evt"].astype(int) - 1
        mom = np.asarray(p.momentum)[hep_rows, 0]
        e_hep = np.sqrt(np.sum(mom ** 2, axis=-1) + 0.511 ** 2) / 1000.0
        if not np.allclose(e_hep, d["ev_eE"], rtol=1e-4):
            raise SystemExit(f"{sf}: stock events do not match the HepEVT events of {hf}")
        s = selected(d)
        for row in np.flatnonzero(s):
            edep, evis = stock_cells(d, row)
            vp = int(plane_of(d["ev_vz"][row], pe_edges))
            stock_obs.append(l1_event(edep, evis, d["ev_eE"][row] * 1000.0, vp, view))
            vtx_planes.append(vp)
        sel_parts.append(jax.tree_util.tree_map(lambda x: np.asarray(x)[hep_rows][s], p))
        n_ev += int(s.sum())
    parts = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *sel_parts)
    rows = list(range(n_ev))

    ref = a.out / "reference" / "D1-L1"
    (ref / "stock").mkdir(parents=True, exist_ok=True)
    (ref / "stock" / "hists.json").write_text(json.dumps(hists_l1(stock_obs)))
    (ref / "reference.json").write_text(json.dumps({"stock_rate_per_slot": None, "mode": "same_events",
                                                    "boundary": "HepEVT electron -> per-cell scintillator deposits",
                                                    "split": a.split}))
    for dt in DTYPES:
        obs = detsim_l1(det, params, parts, rows, dt, a.batch, vtx_planes, view, a.seed if a.sampled else None)
        sub = a.out / "submission" / "D1-L1" / dt
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "hists.json").write_text(json.dumps(hists_l1(obs)))
    res = score(a.out / "submission", a.out / "reference")
    (a.out / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    rec = {"criteria_sha256": hashlib.sha256((HERE / "criteria.json").read_bytes()).hexdigest(),
           "split": a.split, "events": n_ev, "fits": deposit_fit,
           "deposits": "sampled (one draw per event, seed %d)" % a.seed if a.sampled else "expected", "chunks": [c for c in ids if use(c)],
           "L1": {dt: {h["hist"]: {k: h[k] for k in ("chi2ndf", "ndf", "p", "flag")} for h in r["hists"]}
                  for dt, r in res["workflows"]["D1-L1"].items()}}
    (a.out / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
