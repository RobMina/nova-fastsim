"""D1 stock-deposit replay (benchmarks/d1/replay_criteria.json): stock FLS deposits -> world Segments -> detsim cell
response and front end.

    python benchmarks/d1/d1_replay.py signs   OUT_DIR --description ND.json --fls CHUNK_fls.npz ... --chunk-ids ...
    python benchmarks/d1/d1_replay.py closure OUT_DIR --description ND.json --fls ... --chunk-ids ... --signs SIGNS.json
    python benchmarks/d1/d1_replay.py l2      OUT_DIR --description ND.json --fls ... --stock CHUNK.npz ...
        --hepevt CHUNK.txt ... --chunk-ids ... --signs SIGNS.json --params FIT.json ... [--fit-max-adc A] [--fit-quiet-mev E]

``--fls`` files come from d1_fls_to_npz.py and ``--stock`` files from the private stock converter, one per chunk, in
the same order (both follow the dump's event order). Stock data and results are collaboration data: point OUT_DIR at
a private area. Needs JAX, hep-simbench and the ND description.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_compare as D  # noqa: E402
import d1_l2 as L2  # noqa: E402

CRIT = HERE / "replay_criteria.json"
DTYPES = ("float32", "float64")
MIN_LEN, POINT_HALF = 1e-4, 1e-3


# ------------------------------------------------------------------------------------------- frame

def description_arrays(path):
    desc = json.loads(Path(path).read_text())
    pl = desc["planes"]
    k = desc["constants"]
    return {"z": np.array([p["z"] for p in pl]), "view": np.array([p["view"] for p in pl], int),
            "u": np.array([p["u"] for p in pl]), "lc": np.array([p["long_centre"] for p in pl]),
            "depth": float(k["cell_depth"])}


def to_world(g, plane, cell, X, Y, Z, signs):
    """World points ``[n, 3]`` of local points; ``signs[view] = (s_u, s_z, s_w)``. Plane z is mid-depth."""
    view = g["view"][plane]
    s = np.array([signs[0], signs[1]])[view]                        # [n, 3]
    u = g["u"][plane, cell] + s[:, 0] * X
    z = g["z"][plane] + s[:, 1] * Y
    w = g["lc"][plane] + s[:, 2] * Z
    x = np.where(view == 0, u, w)
    y = np.where(view == 0, w, u)
    return np.stack([x, y, z], axis=-1)


def segments_world(g, f, sel, signs):
    """p0, p1 [n, 3] (cm), edep (MeV), t (ns) for FLS rows ``sel``; point-like pieces get 2e-3 cm along the cell."""
    pl, ce = f["fls_plane"][sel], f["fls_cell"][sel]
    p0 = to_world(g, pl, ce, f["fls_x0"][sel], f["fls_y0"][sel], f["fls_z0"][sel], signs)
    p1 = to_world(g, pl, ce, f["fls_x1"][sel], f["fls_y1"][sel], f["fls_z1"][sel], signs)
    short = np.linalg.norm(p1 - p0, axis=-1) < MIN_LEN
    if short.any():
        view = g["view"][pl[short]]
        ax = np.where(view[:, None] == 0, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        mid = 0.5 * (p0[short] + p1[short])
        p0[short], p1[short] = mid - POINT_HALF * ax, mid + POINT_HALF * ax
    return p0, p1, f["fls_edep"][sel] * 1000.0, f["fls_t"][sel]


def fit_signs(g, fls_list):
    """Per view, the sign triple minimising the median distance of early primary deposits to the electron line."""
    table = {0: {}, 1: {}}
    combos = list(itertools.product((1, -1), repeat=3))
    pts = {0: [], 1: []}
    for f in fls_list:
        sel = D.selected({k: f[k] for k in f if k.startswith("ev_")}) if "ev_eE" in f else np.ones(len(f["ev_evt"]), bool)
        rows = np.flatnonzero(sel)
        keep = np.isin(f["fls_row"], rows) & (np.abs(f["fls_pdg"]) == 11) & (f["fls_plane"] < len(g["z"]))
        r = f["fls_row"][keep]
        v = np.c_[f["ev_vx"][r], f["ev_vy"][r], f["ev_vz"][r]]
        d = np.c_[f["ev_epx"][r], f["ev_epy"][r], f["ev_epz"][r]]
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        dz = g["z"][f["fls_plane"][keep]] - v[:, 2]
        near = (dz > -3.0) & (dz < 30.0 * np.maximum(d[:, 2], 0.05))
        idx = np.flatnonzero(keep)[near]
        for vw in (0, 1):
            m = g["view"][f["fls_plane"][idx]] == vw
            pts[vw].append((idx[m], v[near][m], d[near][m], f))
    for vw in (0, 1):
        for c in combos:
            dist = []
            for idx, v, d, f in pts[vw]:
                signs = {0: c, 1: c}
                pl, ce = f["fls_plane"][idx], f["fls_cell"][idx]
                mid = 0.5 * (to_world(g, pl, ce, f["fls_x0"][idx], f["fls_y0"][idx], f["fls_z0"][idx], signs)
                             + to_world(g, pl, ce, f["fls_x1"][idx], f["fls_y1"][idx], f["fls_z1"][idx], signs))
                rel = mid - v
                dist.append(np.linalg.norm(rel - (rel * d).sum(1, keepdims=True) * d, axis=1))
            table[vw][str(c)] = float(np.median(np.concatenate(dist)))
    best = {vw: [int(x) for x in min(table[vw], key=table[vw].get).strip("()").split(",")] for vw in (0, 1)}
    return best, table, {vw: int(sum(len(p[0]) for p in pts[vw])) for vw in (0, 1)}


# ------------------------------------------------------------------------------------------- replay

def batches(g, f, rows, signs, dtype, size=25):
    """Padded segment batches ``(rows, Segments [B, S])`` for the event rows of one FLS file."""
    import jax.numpy as jnp
    from novafastsim_detsim import deposits
    order = np.argsort(f["fls_row"], kind="stable")
    fr = f["fls_row"][order]
    for i in range(0, len(rows), size):
        rb = rows[i:i + size]
        segs = []
        for r in rb:
            sel = order[np.searchsorted(fr, r, "left"):np.searchsorted(fr, r, "right")]
            sel = sel[f["fls_plane"][sel] < len(g["z"])]
            segs.append(segments_world(g, f, sel, signs))
        S = max(256, int(np.ceil(max(len(s[2]) for s in segs) / 256.0)) * 256)
        B = len(rb)
        p0, p1 = np.zeros((B, S, 3)), np.zeros((B, S, 3))
        e, t, m = np.zeros((B, S)), np.zeros((B, S)), np.zeros((B, S), bool)
        for b, (a0, a1, ee, tt) in enumerate(segs):
            n = len(ee)
            p0[b, :n], p1[b, :n], e[b, :n], t[b, :n], m[b, :n] = a0, a1, ee, tt, True
        p1 = np.where(m[..., None], p1, p0 + np.array([0.0, 0.0, POINT_HALF]))
        dt = getattr(jnp, dtype)
        yield rb, deposits.Segments(jnp.asarray(p0, dt), jnp.asarray(p1, dt), jnp.asarray(e, dt), jnp.asarray(t, dt),
                                    jnp.asarray(t, dt), jnp.full((B, S), deposits.TRACK, jnp.int32), jnp.asarray(m))


def stock_cells_fls(f, r, P, C):
    s = (f["fls_row"] == r) & (f["fls_plane"] < P)
    e, eb = np.zeros((P, C)), np.zeros((P, C))
    np.add.at(e, (f["fls_plane"][s], f["fls_cell"][s]), f["fls_edep"][s] * 1000.0)
    np.add.at(eb, (f["fls_plane"][s], f["fls_cell"][s]), f["fls_edep_birks"][s] * 1000.0)
    return e, eb


# ------------------------------------------------------------------------------------------- modes

def run_signs(a, g):
    fls = [dict(np.load(p)) for c, p in zip(a.chunk_ids, a.fls) if c < 10]
    best, table, n = fit_signs(g, fls)
    out = {"criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(), "signs": best,
           "median_distance_cm": table, "deposits_used": n}
    (a.out / "signs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def load_signs(path):
    s = json.loads(Path(path).read_text())["signs"]
    return {int(k): tuple(v) for k, v in s.items()}


def run_closure(a, g, det, params):
    import jax
    from novafastsim_detsim import config, sim
    signs = load_signs(a.signs)
    P, C = det.geometry.n_planes, det.geometry.n_cells
    acc = jax.jit(jax.vmap(lambda s: sim.accumulate(det, pp, s)))
    res = {}
    for dt in DTYPES:
        pp = config.to_device(params, getattr(jax.numpy, dt))
        own, tot, bk_r, bk_s = [], [], [], []
        for cid, path in zip(a.chunk_ids, a.fls):
            if cid < 10:
                continue
            f = dict(np.load(path))
            rows = np.flatnonzero(D.selected({k: f[k] for k in f if k.startswith("ev_")}))
            for rb, seg in batches(g, f, rows, signs, dt):
                r = acc(seg)
                ed, ev = np.asarray(r.edep, np.float64), np.asarray(r.evis, np.float64)
                for b, row in enumerate(rb):
                    se, sb = stock_cells_fls(f, row, P, C)
                    if ed[b].sum() <= 0 or se.sum() <= 0:
                        continue
                    own.append(np.minimum(ed[b], se).sum() / ed[b].sum())
                    tot.append(ed[b].sum() / se.sum())
                    bk_r.append(ev[b].sum() / ed[b].sum())
                    bk_s.append(sb.sum() / se.sum())
        own, tot = np.array(own), np.array(tot)
        res[dt] = {"events": int(len(own)), "own_cell_fraction_median": float(np.median(own)),
                   "own_cell_fraction_q01": float(np.quantile(own, 0.01)), "total_ratio_median": float(np.median(tot)),
                   "total_ratio_q01_q99": np.quantile(tot, [0.01, 0.99]).tolist(),
                   "birks_ratio_replay_median": float(np.median(bk_r)), "birks_ratio_stock_median": float(np.median(bk_s)),
                   "pass": bool(np.median(own) > 0.99 and abs(np.median(tot) - 1.0) < 0.01)}
    out = {"criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(), "signs": a.signs.name,
           "own_cell_definition": "cell-level overlap sum_c min(replay_c, stock_c) / sum_c replay_c", "closure": res}
    (a.out / "closure.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def run_l2(a, g, det, params):
    import jax
    import jax.numpy as jnp
    from hepsimbench.score import score
    from scintfibrefastsim.backends import nova_apd
    from novafastsim_detsim import config, sim
    from novafastsim_detsim.geometry import layout
    signs = load_signs(a.signs)
    det, fits = D.apply_fit(det, params, a.params)
    gg = det.geometry
    lay = layout(det, jnp.float64)
    geo = {"z": np.asarray(lay.z_lo) + 0.5 * gg.cell_depth, "view": np.asarray(lay.view),
           "wc": np.asarray(lay.w_centre), "length": gg.cell_length, "readout": gg.readout_end}
    grp = det.settings.apd_group or 32
    st_train, _ = L2.load(a, "train")
    h_tr, e_tr = L2.stock_hits(st_train, geo, grp)
    h_fit = h_tr if a.fit_max_adc is None else h_tr[h_tr[:, 4] < a.fit_max_adc]
    if a.fit_quiet_mev is not None:
        h_fit = h_fit[h_fit[:, 5] < a.fit_quiet_mev]
    R, E, _ = L2.ratio_profile(h_fit, L2.FIT_BINS, len(e_tr), L2.FIT_MIN_MEV)
    centres = 0.5 * (L2.FIT_BINS[1:] + L2.FIT_BINS[:-1])
    optics = L2.fit_optics(det, params, R, E, centres)
    params.update({k: v for k, v in optics.items() if k.startswith("optics.")})
    pe_frac = e_tr[:, 1] / e_tr[:, 2]
    rng = (0.0, float(np.quantile(pe_frac, 0.995)) * 1.1)

    st_test, _ = L2.load(a, "test")
    h_te, e_te = L2.stock_hits(st_test, geo, grp)
    ref = a.out / "reference" / "D1-L2R"
    (ref / "stock").mkdir(parents=True, exist_ok=True)
    (ref / "stock" / "hists.json").write_text(json.dumps(L2.l2_hists(h_te, e_te, rng)))
    (ref / "reference.json").write_text(json.dumps({"stock_rate_per_slot": None, "mode": "same_events", "split": "test",
                                                    "boundary": "stock FLS deposits -> signal cell hits (PE)"}))
    test_fls = [(cid, p) for cid, p in zip(a.chunk_ids, a.fls) if cid >= 10]
    groups = sim.apd_groups(det)
    planes = np.repeat(np.arange(gg.n_planes), gg.n_cells).reshape(gg.n_planes, gg.n_cells)
    birks = {}
    for dt in DTYPES:
        pp = config.to_device(params, getattr(jnp, dt))

        @jax.jit
        def respond(seg, keys):
            def one(s, k):
                r = sim.accumulate(det, pp, s)
                h = nova_apd.sample(k, r.pe, r.time, config.apd(pp), det.settings.readout(), groups)
                return r.evis, r.edep, h[0], h[3]
            return jax.vmap(one)(seg, keys)

        hits, evs, ev_i, bk = [], [], 0, []
        for (cid, path), (d, rows) in zip(test_fls, st_test):
            f = dict(np.load(path))
            for rb, seg in batches(g, f, rows, signs, dt):
                keys = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(a.seed), j))(
                    jnp.asarray(cid * 100000 + rb))
                evis, edep, pe, hit = (np.asarray(x) for x in respond(seg, keys))
                for b, row in enumerate(rb):
                    sig = hit[b] & (evis[b] > 0)
                    pl = planes[sig]
                    vw = geo["view"][pl]
                    vtx = np.array([d["ev_vx"][row], d["ev_vy"][row], d["ev_vz"][row]])
                    mom = np.array([d["ev_epx"][row], d["ev_epy"][row], d["ev_epz"][row]])
                    w = L2.axis_w(vtx, mom, geo["z"][pl], vw)
                    hits.append(np.c_[np.full(sig.sum(), ev_i), pe[b][sig].astype(np.float64), evis[b][sig],
                                      L2.distance(geo, pl, vw, w)])
                    evs.append((sig.sum(), pe[b][sig].sum(), d["ev_eE"][row], 0, 0.0))
                    if edep[b].sum() > 0:
                        bk.append(evis[b].sum() / edep[b].sum())
                    ev_i += 1
        birks[dt] = float(np.median(bk))
        sub = a.out / "submission" / "D1-L2R" / dt
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "hists.json").write_text(json.dumps(L2.l2_hists(np.concatenate(hits), np.array(evs), rng)))
    res = score(a.out / "submission", a.out / "reference")
    (a.out / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    rec = {"criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(), "signs": a.signs.name, "fits": fits,
           "optics": optics, "fit_max_adc": a.fit_max_adc, "fit_quiet_mev": a.fit_quiet_mev, "seed": a.seed,
           "test_events": int(len(e_te)), "birks_ratio_replay_median": birks,
           "L2R": {dt: {h["hist"]: {k: h[k] for k in ("chi2ndf", "ndf", "p", "flag")} for h in r["hists"]}
                   for dt, r in res["workflows"]["D1-L2R"].items()}}
    (a.out / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from novafastsim_detsim import config
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("signs", "closure", "l2"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--fls", type=Path, nargs="+", required=True)
    ap.add_argument("--stock", type=Path, nargs="*", default=[])
    ap.add_argument("--hepevt", type=Path, nargs="*", default=[])
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--signs", type=Path)
    ap.add_argument("--params", type=Path, nargs="*", default=[])
    ap.add_argument("--fit-max-adc", type=float)
    ap.add_argument("--fit-quiet-mev", type=float)
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args(argv)
    if len(a.fls) != len(a.chunk_ids):
        raise SystemExit("--fls and --chunk-ids need the same length")
    a.out.mkdir(parents=True, exist_ok=True)
    g = description_arrays(a.description)
    det, params = config.load_description(a.description)
    if a.mode == "signs":
        run_signs(a, g)
    elif a.mode == "closure":
        run_closure(a, g, det, params)
    else:
        if not (len(a.stock) == len(a.hepevt) == len(a.chunk_ids)):
            raise SystemExit("l2 needs --stock and --hepevt per chunk")
        run_l2(a, g, det, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
