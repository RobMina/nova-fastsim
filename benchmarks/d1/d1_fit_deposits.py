"""D1 deposit-model fit on the train chunks (benchmarks/d1/deposit_fit_criteria.json).

    python benchmarks/d1/d1_fit_deposits.py OUT.json --description ND.json --stock CHUNK.npz ... --hepevt CHUNK.txt ...
        --chunk-ids ID ... [--max-events 2500] [--batch 50]

Fits the shower and electron-head parameters of the deposit model to the stock per-cell deposits of the same HepEVT
electrons: the per-plane longitudinal profile, a soft-binned lateral profile and the scintillator fraction, by least
squares with the Jacobian from ``jax.jacfwd`` through the detector simulation (float64). Only train chunks (ids 0-9)
are accepted. OUT.json holds the fitted values as a ``params`` override dict, read by ``d1_compare.py --params`` and
``d1_l2.py --params``. Stock data and results are collaboration data: write OUT.json to a private area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d1_compare import N_LAT, N_LONG, plane_of, selected, stock_cells  # noqa: E402

NAMES = ("head_x0", "shower_tmax_shift", "shower_b", "lat_core", "lat_tail", "lat_pcore", "shower_inert_scale")


def bounds():
    c = json.loads((HERE / "deposit_fit_criteria.json").read_text())["fit"]["bounds"]
    return np.array([c[n][0] for n in NAMES], float), np.array([c[n][1] for n in NAMES], float)


def soft_lat(xp, edep, view, C):
    """Lateral profile with linear soft binning: ``edep [..., P, C]`` -> ``[..., N_LAT]`` (unnormalised)."""
    cells = xp.arange(C, dtype=edep.dtype)
    k = xp.arange(N_LAT, dtype=edep.dtype)
    out = 0.0
    for v in (0, 1):
        e = edep[..., view == v, :].sum(-2)                              # [..., C]: the distance depends on the cell only
        tot = e.sum(-1)
        pos = tot > 0
        mean = (e * cells).sum(-1) / xp.where(pos, tot, 1.0)
        x = xp.minimum(xp.abs(cells - mean[..., None]), N_LAT - 1.0)
        w = xp.maximum(1.0 - xp.abs(x[..., None] - k), 0.0)               # [..., C, N_LAT]
        out = out + (e[..., None] * w).sum(-2)
    return out


def observables(xp, edep, e_true, vtx_plane, view):
    """Per-event observable vector ``[..., N_LONG + N_LAT + 1]`` (fractions of E_true)."""
    P, C = edep.shape[-2:]
    per_plane = edep.sum(-1)                                            # [..., P]
    k = vtx_plane[..., None] + xp.arange(N_LONG)
    ok = k < P
    lp = xp.where(ok, xp.take_along_axis(per_plane, xp.minimum(k, P - 1), axis=-1), 0.0)
    lat = soft_lat(xp, edep, view, C)
    esc = edep.sum(axis=(-1, -2))[..., None]
    return xp.concatenate([lp, lat, esc], axis=-1) / e_true[..., None]


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import least_squares
    from novafastsim_detsim import config, particles, sim
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-events", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args(argv)
    if not (len(a.stock) == len(a.hepevt) == len(a.chunk_ids)):
        raise SystemExit("--stock, --hepevt and --chunk-ids need the same length")
    if any(c >= 10 for c in a.chunk_ids):
        raise SystemExit("the deposit fit uses train chunks (ids 0-9) only")

    det, params = config.load_description(a.description)
    lay = layout(det)
    pe_edges = np.asarray(lay.plane_edges, np.float64)
    view = np.asarray(lay.view)
    stock_obs, parts_l, vtx, etrue = [], [], [], []
    for sf, hf in zip(a.stock, a.hepevt):
        d = dict(np.load(sf))
        p = particles.from_hepevt(hf.read_text())
        hep_rows = d["ev_evt"].astype(int) - 1
        mom = np.asarray(p.momentum)[hep_rows, 0]
        if not np.allclose(np.sqrt(np.sum(mom ** 2, -1) + 0.511 ** 2) / 1000.0, d["ev_eE"], rtol=1e-4):
            raise SystemExit(f"{sf}: stock events do not match the HepEVT events of {hf}")
        s = selected(d)
        for row in np.flatnonzero(s):
            edep, _ = stock_cells(d, row)
            vp = int(plane_of(d["ev_vz"][row], pe_edges))
            stock_obs.append(observables(np, edep, np.float64(d["ev_eE"][row] * 1000.0), np.int64(vp), view))
            vtx.append(vp)
            etrue.append(d["ev_eE"][row] * 1000.0)
        parts_l.append(jax.tree_util.tree_map(lambda x: np.asarray(x)[hep_rows][s], p))
    parts = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *parts_l)
    n_all = len(vtx)
    take = np.arange(n_all)[::max(1, int(np.ceil(n_all / a.max_events)))]
    S = np.asarray(stock_obs)[take]
    vtx, etrue = np.asarray(vtx)[take], np.asarray(etrue)[take]
    parts = jax.tree_util.tree_map(lambda x: np.asarray(x)[take], parts)
    target, se_s = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))

    base = config.to_device(params, jnp.float64)

    @jax.jit
    def batch_sum(theta, sub, vp, et):
        p = dict(base)
        for i, n in enumerate(NAMES):
            p[n] = theta[i]
        r = sim.expected_batch(det, p, sub)
        return observables(jnp, r.edep, et, vp, view).sum(0)

    jac_sum = jax.jit(jax.jacfwd(batch_sum))
    batches = []
    for i in range(0, len(take), a.batch):
        sl = slice(i, i + a.batch)
        sub = jax.tree_util.tree_map(lambda x: jnp.asarray(x[sl], jnp.float64)
                                     if np.issubdtype(x.dtype, np.floating) else jnp.asarray(x[sl]), parts)
        batches.append((sub, jnp.asarray(vtx[sl]), jnp.asarray(etrue[sl], jnp.float64)))

    @jax.jit
    def batch_sq(theta, sub, vp, et):
        p = dict(base)
        for i, n in enumerate(NAMES):
            p[n] = theta[i]
        r = sim.expected_batch(det, p, sub)
        return (observables(jnp, r.edep, et, vp, view) ** 2).sum(0)

    # errors: stock and detsim event-to-event spreads combined (detsim's at the neutral start point, then fixed)
    x_start = np.array([0.0, 0.0, float(params["shower_b"]), 1.0, 1.0, 1.0, 1.0])
    t_s = jnp.asarray(x_start, jnp.float64)
    m1 = np.asarray(sum(batch_sum(t_s, *b) for b in batches)) / len(take)
    m2 = np.asarray(sum(batch_sq(t_s, *b) for b in batches)) / len(take)
    se_d = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) * len(take) / (len(take) - 1) / len(take))
    se = np.sqrt(se_s ** 2 + se_d ** 2)
    use = se > 0
    print(f"fit events {len(take)} of {n_all}; residual bins {int(use.sum())}", flush=True)

    cache = {}

    def evaluate(theta):
        key = tuple(np.round(theta, 12))
        if key not in cache:
            t = jnp.asarray(theta, jnp.float64)
            f = sum(batch_sum(t, *b) for b in batches) / len(take)
            J = sum(jac_sum(t, *b) for b in batches) / len(take)
            cache.clear()
            cache[key] = (np.asarray(f), np.asarray(J))
        return cache[key]

    def resid(theta):
        f, _ = evaluate(theta)
        r = ((f - target) / np.where(use, se, 1.0))[use]
        print(f"  chi2 {np.sum(r ** 2):10.1f}  theta {np.round(theta, 4).tolist()}", flush=True)
        return r

    def jac(theta):
        _, J = evaluate(theta)
        return (J / np.where(use, se, 1.0)[:, None])[use]

    t0 = time.time()
    r0 = resid(x_start)
    lo, hi = bounds()
    x_start_in = np.clip(x_start + np.array([0.5, 0, 0, 0, 0, 0, 0]), lo, hi)   # head_x0 starts off the bound
    fit = least_squares(resid, x_start_in, jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=60)
    f, _ = evaluate(fit.x)
    chi2_parts = {}
    for name, s_ in (("long_profile", slice(0, N_LONG)), ("lat_profile", slice(N_LONG, N_LONG + N_LAT)),
                     ("escint_frac", slice(N_LONG + N_LAT, None))):
        rr = ((f - target) / np.where(use, se, 1.0))[s_][use[s_]]
        chi2_parts[name] = {"chi2": float(np.sum(rr ** 2)), "bins": int(use[s_].sum())}
    J = jac(fit.x)
    cov = np.linalg.pinv(J.T @ J)
    out = {
        "criteria_sha256": hashlib.sha256((HERE / "deposit_fit_criteria.json").read_bytes()).hexdigest(),
        "chunks": a.chunk_ids, "events": int(len(take)), "events_available": n_all,
        "params": {n: float(v) for n, v in zip(NAMES, fit.x)},
        "errors": {n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(NAMES)},
        "chi2_start": float(np.sum(r0 ** 2)), "chi2": float(np.sum(fit.fun ** 2)), "bins": int(use.sum()),
        "chi2_parts": chi2_parts, "nfev": int(fit.nfev), "status": int(fit.status), "message": fit.message,
        "seconds": time.time() - t0,
        "means": {"escint_frac": {"stock": float(target[-1]), "detsim": float(f[-1])}},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
