"""D1 front-end crosstalk: train fit of apd.crosstalk (benchmarks/d1/crosstalk_criteria.json).

    python benchmarks/d1/d1_fit_crosstalk.py OUT.json --description ND.json --params DEP.json STOCH.json SAT.json
        --stock CHUNK.npz ... --hepevt CHUNK.txt ... --chunk-ids ID ... [--max-events 1000] [--seed 17]

Only train chunks (ids 0-9) are accepted. OUT.json holds ``settings`` (``apd_group``) and ``params``
(``apd.crosstalk``) for ``d1_l2.py --params``, applied after the earlier fits (the L2 run uses --fit-max-adc 3000,
as the saturation fit's optics did). Stock data and results are collaboration data: write OUT.json to a private area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_compare as D  # noqa: E402
import d1_l2 as L2  # noqa: E402

CRIT = HERE / "crosstalk_criteria.json"
E_BANDS = np.array([1.5, 3.0, 6.0, 12.0])
O_BINS = np.array([0.0, 20.0, 60.0, 150.0, 400.0, 2000.0])


def group_others(e, groups):
    tot = np.bincount(groups.ravel(), weights=e.ravel(), minlength=groups.size)
    return tot[groups] - e


def accumulate(counts, e, hit, others):
    """Add one event's cells (``e`` [P, C] visible MeV, ``hit`` bool, ``others`` MeV) to ``counts`` [2, 3, 5]."""
    ib = np.digitize(e, E_BANDS) - 1
    io = np.digitize(others, O_BINS) - 1
    ok = (ib >= 0) & (ib < len(E_BANDS) - 1) & (io >= 0) & (io < len(O_BINS) - 1)
    np.add.at(counts[0], (ib[ok], io[ok]), 1)
    np.add.at(counts[1], (ib[ok], io[ok]), hit[ok])


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import minimize_scalar
    from novafastsim_detsim import config, sim

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="*", default=[])
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-events", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    if not (len(a.stock) == len(a.hepevt) == len(a.chunk_ids)):
        raise SystemExit("--stock, --hepevt and --chunk-ids need the same length")
    if any(c >= 10 for c in a.chunk_ids):
        raise SystemExit("the crosstalk fit uses train chunks (ids 0-9) only")
    crit = json.loads(CRIT.read_text())
    fc = crit["fit_on_train"]
    t0 = time.time()

    det, params = config.load_description(a.description)
    det, prior = D.apply_fit(det, params, a.params)
    det = replace(det, settings=replace(det.settings, **crit["settings"]))
    for f in a.params:                                   # the optics the saturation fit used (ADC-cut level-2 fit)
        opt = json.loads(f.read_text()).get("optics_used")
        if opt:
            params.update({k: v for k, v in opt.items() if k.startswith("optics.")})
    P, C = det.geometry.n_planes, det.geometry.n_cells
    groups = np.asarray(sim.apd_groups(det))

    st_train, parts = L2.load(a, "train")
    n_all = np.asarray(parts.pdg).shape[0]
    take = np.arange(n_all)[::max(1, int(np.ceil(n_all / a.max_events)))]

    # ---- stock efficiency table on the same events
    stock = np.zeros((2, len(E_BANDS) - 1, len(O_BINS) - 1))
    k = 0
    pick = set(take.tolist())
    for d, rows in st_train:
        sig_keys = None
        for row in rows:
            if k in pick:
                if sig_keys is None:
                    s = (d["hit_true_e"] > 0) & (d["hit_plane"] < P)
                    sig_keys = set((d["hit_row"][s].astype(np.int64) * 10 ** 6 + d["hit_plane"][s] * 1000
                                    + d["hit_cell"][s]).tolist())
                m = (d["dep_row"] == row) & (d["dep_plane"] < P)
                e = np.zeros((P, C)); hit = np.zeros((P, C), bool)
                e[d["dep_plane"][m], d["dep_cell"][m]] = d["dep_edep_birks"][m] * 1000.0
                keys = row * 10 ** 6 + d["dep_plane"][m] * 1000 + d["dep_cell"][m]
                hit[d["dep_plane"][m], d["dep_cell"][m]] = [x in sig_keys for x in keys.tolist()]
                accumulate(stock, e, hit, group_others(e, groups))
            k += 1
    sub = jax.tree_util.tree_map(lambda x: np.asarray(x)[take], parts)

    def detsim_table(kappa):
        p = dict(params)
        p["apd.crosstalk"] = kappa
        pp = config.to_device(p, jnp.float64)
        key = jax.random.PRNGKey(a.seed)
        tab = np.zeros_like(stock)
        n = len(take)
        for i in range(0, n, 50):
            b = jax.tree_util.tree_map(lambda x: jnp.asarray(x[i:i + 50], jnp.float64)
                                       if np.issubdtype(x.dtype, np.floating) else jnp.asarray(x[i:i + 50]), sub)
            r, h = sim.sampled_batch(det, pp, b, jax.random.fold_in(key, i))
            ev, hh = np.asarray(r.evis, np.float64), np.asarray(h.hit)
            for j in range(ev.shape[0]):
                accumulate(tab, ev[j], hh[j] & (ev[j] > 0), group_others(ev[j], groups))
        return tab

    def eff(t):
        n = np.maximum(t[0], 1.0)
        e = t[1] / n
        return e, e * (1 - e) / n

    es, vs = eff(stock)

    def chi2(kappa):
        t = detsim_table(kappa)
        ed, vd = eff(t)
        ok = (stock[0] >= 20) & (t[0] >= 20)
        c = float(np.sum(((ed - es) ** 2 / np.maximum(vs + vd, 1e-12))[ok]))
        print(f"  crosstalk {kappa:.5f}: chi2 {c:9.1f} over {int(ok.sum())} points", flush=True)
        return c

    lo, hi = fc["bounds"]
    c0 = chi2(0.0)
    res = minimize_scalar(chi2, bounds=(lo, hi), method="bounded", options={"xatol": 2e-4})
    t_fin = detsim_table(float(res.x))
    ed, _ = eff(t_fin)
    out = {
        "criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(),
        "prior_fits": prior, "chunks": a.chunk_ids, "events": int(len(take)), "seed": a.seed,
        "settings": crit["settings"], "params": {"apd.crosstalk": float(res.x)},
        "chi2_no_crosstalk": c0, "chi2": float(res.fun), "nfev": int(res.nfev),
        "table": {"e_bands_mev": E_BANDS.tolist(), "other_bins_mev": O_BINS.tolist(),
                  "stock_eff": es.tolist(), "stock_cells": stock[0].tolist(), "detsim_eff": ed.tolist(),
                  "detsim_cells": t_fin[0].tolist()},
        "seconds": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("params", "chi2_no_crosstalk", "chi2", "nfev", "seconds")}, indent=1))
    np.set_printoptions(precision=3, suppress=True)
    print("stock eff\n", es, "\ndetsim eff\n", ed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
