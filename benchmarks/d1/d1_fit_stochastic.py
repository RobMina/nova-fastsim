"""D1 stochastic deposit tier: train fits of the readout threshold, the stochastic shower parameters and the Birks
dE/dx of the mean shower (benchmarks/d1/stochastic_criteria.json).

    python benchmarks/d1/d1_fit_stochastic.py OUT.json --description ND.json --params DEPOSIT_FIT.json
        --stock CHUNK.npz ... --hepevt CHUNK.txt ... --chunk-ids ID ... [--max-events 2500] [--batch 25] [--seed 3]

Only train chunks (ids 0-9) are accepted. OUT.json holds ``settings`` (``n_stoch``) and a ``params`` override dict
for ``d1_compare.py --params`` and ``d1_l2.py --params``, applied after the deposit fit. Stock data and results are
collaboration data: write OUT.json to a private area.
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
from d1_compare import apply_fit, plane_of, selected, stock_cells  # noqa: E402
from d1_fit_deposits import observables  # noqa: E402

CRIT = HERE / "stochastic_criteria.json"
N_SPEC, SPEC_LO, SPEC_W = 30, -1.0, 0.1          # log10(MeV) bins over [-1, 2]


def spectrum_np(edep):
    e = edep.ravel()
    e = e[e > 0]
    f = (np.log10(e) - SPEC_LO) / SPEC_W - 0.5
    i0 = np.floor(f).astype(int)
    fr = f - i0
    h = np.zeros(N_SPEC)
    for i, w in ((i0, 1.0 - fr), (i0 + 1, fr)):
        ok = (i >= 0) & (i < N_SPEC)
        np.add.at(h, i[ok], w[ok])
    return h


def spectrum_jnp(edep):
    import jax.numpy as jnp
    e = edep.reshape(-1)
    pos = e > 0
    f = (jnp.log10(jnp.where(pos, e, 1.0)) - SPEC_LO) / SPEC_W - 0.5
    i0 = jnp.floor(f).astype(jnp.int32)
    fr = f - i0
    h = jnp.zeros(N_SPEC, e.dtype)
    for i, w in ((i0, 1.0 - fr), (i0 + 1, fr)):
        ok = pos & (i >= 0) & (i < N_SPEC)
        h = h.at[jnp.clip(i, 0, N_SPEC - 1)].add(jnp.where(ok, w, 0.0))
    return h


def depth_moments(xp, edep, vtx_plane):
    per_plane = edep.sum(-1)
    z = xp.arange(per_plane.shape[-1]) - vtx_plane[..., None]
    tot = per_plane.sum(-1)
    d = (per_plane * z).sum(-1) / xp.where(tot > 0, tot, 1.0)
    return xp.stack([d, d * d], axis=-1)


def full_obs(xp, edep, e_true, vtx_plane, view, spec):
    return xp.concatenate([spec, observables(xp, edep, e_true, vtx_plane, view), depth_moments(xp, edep, vtx_plane)],
                          axis=-1)


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import brentq, least_squares
    from novafastsim_detsim import config, particles, sim
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="*", default=[], help="earlier fit outputs (the deposit fit)")
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-events", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    if not (len(a.stock) == len(a.hepevt) == len(a.chunk_ids)):
        raise SystemExit("--stock, --hepevt and --chunk-ids need the same length")
    if any(c >= 10 for c in a.chunk_ids):
        raise SystemExit("the stochastic fits use train chunks (ids 0-9) only")
    crit = json.loads(CRIT.read_text())
    fc = crit["fits_on_train"]["stochastic"]
    names = fc["params"]

    det, params = config.load_description(a.description)
    det, prior = apply_fit(det, params, a.params)
    from dataclasses import replace
    det = replace(det, settings=replace(det.settings, **crit["model"]["settings"]))
    lay = layout(det)
    pe_edges = np.asarray(lay.plane_edges, np.float64)
    view = np.asarray(lay.view)

    # ---- inputs (train), stock observables, stock signal-hit PE for the threshold
    S_obs, birks_s, parts_l, vtx, etrue, hit_pe = [], [], [], [], [], []
    for sf, hf in zip(a.stock, a.hepevt):
        d = dict(np.load(sf))
        p = particles.from_hepevt(hf.read_text())
        hep_rows = d["ev_evt"].astype(int) - 1
        mom = np.asarray(p.momentum)[hep_rows, 0]
        if not np.allclose(np.sqrt(np.sum(mom ** 2, -1) + 0.511 ** 2) / 1000.0, d["ev_eE"], rtol=1e-4):
            raise SystemExit(f"{sf}: stock events do not match the HepEVT events of {hf}")
        s = selected(d)
        for row in np.flatnonzero(s):
            edep, evis = stock_cells(d, row)
            vp = int(plane_of(d["ev_vz"][row], pe_edges))
            et = d["ev_eE"][row] * 1000.0
            S_obs.append(full_obs(np, edep, np.float64(et), np.int64(vp), view, spectrum_np(edep)))
            birks_s.append(evis.sum() / edep.sum() if edep.sum() > 0 else np.nan)
            vtx.append(vp)
            etrue.append(et)
            hs = (d["hit_row"] == row) & (d["hit_plane"] < 192) & (d["hit_true_e"] > 0)
            hit_pe.append(d["hit_pe"][hs])
        parts_l.append(jax.tree_util.tree_map(lambda x: np.asarray(x)[hep_rows][s], p))
    threshold = round(float(np.quantile(np.concatenate(hit_pe), 0.001)), 1)
    parts = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *parts_l)
    n_all = len(vtx)
    take = np.arange(n_all)[::max(1, int(np.ceil(n_all / a.max_events)))]
    S = np.asarray(S_obs)[take]
    birks_target = float(np.nanmean(np.asarray(birks_s)[take]))
    vtx, etrue = np.asarray(vtx)[take], np.asarray(etrue)[take]
    parts = jax.tree_util.tree_map(lambda x: np.asarray(x)[take], parts)
    target, se_s = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
    print(f"train events {len(take)} of {n_all}; threshold {threshold} PE; stock Birks ratio {birks_target:.5f}",
          flush=True)

    base = config.to_device(params, jnp.float64)
    root = jax.random.PRNGKey(a.seed)
    batches = []
    for i in range(0, len(take), a.batch):
        sl = slice(i, i + a.batch)
        sub = jax.tree_util.tree_map(lambda x: jnp.asarray(x[sl], jnp.float64)
                                     if np.issubdtype(x.dtype, np.floating) else jnp.asarray(x[sl]), parts)
        keys = jax.vmap(lambda j: jax.random.fold_in(root, j))(jnp.arange(i, min(i + a.batch, len(take))))
        batches.append((sub, jnp.asarray(vtx[sl]), jnp.asarray(etrue[sl], jnp.float64), keys))

    def response(theta, sub, keys, extra=None):
        p = dict(base)
        for i, n in enumerate(names):
            p[n] = theta[i]
        if extra:
            p.update(extra)
        return jax.vmap(lambda e, k: sim.sampled_deposits(det, p, e, k))(sub, keys)

    @jax.jit
    def batch_obs(theta, sub, vp, et, keys):
        r = response(theta, sub, keys)
        spec = jax.vmap(spectrum_jnp)(r.edep)
        o = full_obs(jnp, r.edep, et, vp, view, spec)
        return o.sum(0), (o * o).sum(0)

    jac_batch = jax.jit(jax.jacfwd(lambda t, *b: batch_obs(t, *b)[0]))
    x0 = np.array([fc["start"][n] for n in names], float)
    m1, m2 = (np.asarray(sum(x[j] for x in (batch_obs(jnp.asarray(x0), *b) for b in batches))) / len(take)
              for j in (0, 1))
    se_d = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) / (len(take) - 1))
    se = np.sqrt(se_s ** 2 + se_d ** 2)
    use = se > 0
    print(f"residual bins {int(use.sum())}", flush=True)

    cache = {}

    def evaluate(theta):
        key = tuple(np.round(theta, 12))
        if key not in cache:
            t = jnp.asarray(theta, jnp.float64)
            f = sum(batch_obs(t, *b)[0] for b in batches) / len(take)
            J = sum(jac_batch(t, *b) for b in batches) / len(take)
            cache.clear()
            cache[key] = (np.asarray(f), np.asarray(J))
        return cache[key]

    def resid(theta):
        f, _ = evaluate(theta)
        r = ((f - target) / np.where(use, se, 1.0))[use]
        print(f"  chi2 {np.sum(r ** 2):10.1f}  theta {np.round(theta, 4).tolist()}", flush=True)
        return r

    def jac(theta):
        return (evaluate(theta)[1] / np.where(use, se, 1.0)[:, None])[use]

    t0 = time.time()
    r0 = resid(x0)
    lo = np.array([fc["bounds"][n][0] for n in names], float)
    hi = np.array([fc["bounds"][n][1] for n in names], float)
    fit = least_squares(resid, x0, jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=60)
    f, _ = evaluate(fit.x)
    parts_chi2 = {}
    for name, sl in (("cell_spectrum", slice(0, N_SPEC)), ("long_profile", slice(N_SPEC, N_SPEC + 120)),
                     ("lat_profile", slice(N_SPEC + 120, N_SPEC + 150)), ("escint", slice(N_SPEC + 150, N_SPEC + 151)),
                     ("depth_moments", slice(N_SPEC + 151, None))):
        rr = ((f - target) / np.where(use, se, 1.0))[sl][use[sl]]
        parts_chi2[name] = {"chi2": float(np.sum(rr ** 2)), "bins": int(use[sl].sum())}
    cov = np.linalg.pinv(jac(fit.x).T @ jac(fit.x))
    depth_sd = {"stock": float(np.sqrt(max(target[-1] - target[-2] ** 2, 0.0))),
                "detsim": float(np.sqrt(max(f[-1] - f[-2] ** 2, 0.0)))}
    print(f"stochastic fit done in {time.time() - t0:.0f} s: {fit.x}", flush=True)

    # ---- Birks: shower_dedx so the mean sampled EdepBirks / Edep equals stock's
    tfit = jnp.asarray(fit.x)

    @jax.jit
    def birks_batch(sd, sub, keys):
        r = response(tfit, sub, keys, {"shower_dedx": sd})
        return jnp.sum(r.evis.sum((-1, -2)) / r.edep.sum((-1, -2)))

    def g(sd):
        v = sum(birks_batch(jnp.asarray(sd, jnp.float64), b[0], b[3]) for b in batches) / len(take)
        print(f"  shower_dedx {sd:.4f}  Birks ratio {float(v):.5f}", flush=True)
        return float(v) - birks_target

    lo_b, hi_b = crit["fits_on_train"]["birks"].split("bounds [")[1].split("]")[0].split(",")
    lo_b, hi_b = float(lo_b), float(hi_b)
    g_lo, g_hi = g(lo_b), g(hi_b)
    if g_lo * g_hi < 0:
        shower_dedx, birks_note = brentq(g, lo_b, hi_b, xtol=1e-3), "root"
    else:
        shower_dedx, birks_note = (lo_b, "at lower bound") if abs(g_lo) < abs(g_hi) else (hi_b, "at upper bound")

    out = {
        "criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(),
        "prior_fits": prior, "chunks": a.chunk_ids, "events": int(len(take)), "events_available": n_all,
        "seed": a.seed,
        "settings": crit["model"]["settings"],
        "params": {**{n: float(v) for n, v in zip(names, fit.x)}, "apd.threshold_pe": threshold,
                   "shower_dedx": float(shower_dedx)},
        "errors": {n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)},
        "chi2_start": float(np.sum(r0 ** 2)), "chi2": float(np.sum(fit.fun ** 2)), "bins": int(use.sum()),
        "chi2_parts": parts_chi2, "nfev": int(fit.nfev), "status": int(fit.status), "message": fit.message,
        "depth_sd_planes": depth_sd, "birks": {"stock": birks_target, "note": birks_note},
        "cell_spectrum": {"centres_log10_mev": (SPEC_LO + SPEC_W * (np.arange(N_SPEC) + 0.5)).tolist(),
                          "stock": target[:N_SPEC].tolist(), "detsim": f[:N_SPEC].tolist()},
        "seconds": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("params", "errors", "chi2_start", "chi2", "bins", "chi2_parts",
                                          "depth_sd_planes", "birks", "message", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
