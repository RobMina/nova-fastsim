"""D1 front end fitted on replayed stock deposits (benchmarks/d1/frontend_replay_criteria.json).

    python benchmarks/d1/d1_replay_frontend.py OUT.json --description ND.json --fls CHUNK_fls.npz ...
        --stock CHUNK.npz ... --hepevt CHUNK.txt ... --chunk-ids ID ... --signs SIGNS.json
        --params DEPOSIT.json STOCHASTIC.json SATURATION.json CROSSTALK.json [--max-events 1000] [--seed 31]

Only train chunks (ids 0-9) are used. OUT.json holds ``params`` (``optics.yield_per_mev``, ``apd.sat_recovery``,
``apd.crosstalk``), ``settings`` (``apd_group``) and ``optics_table`` (the fitted collection curve), to apply after
the earlier fits with ``--no-optics-fit``. Stock data and results are collaboration data: keep OUT.json private.
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
import d1_compare as D  # noqa: E402
import d1_fit_crosstalk as XT  # noqa: E402
import d1_l2 as L2  # noqa: E402
import d1_replay as RP  # noqa: E402

CRIT = HERE / "frontend_replay_criteria.json"
KNOTS_CM = np.arange(0.0, 400.0 + 1e-9, 20.0)
MIN_MEV, QUIET_MEV, MAX_ADC, SAT_MEV = 2.0, 20.0, 3000.0, 100.0
MAX_ROUNDS, TOL = 4, 0.01


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import brentq, minimize_scalar, nnls
    from scintfibrefastsim.backends import nova_apd
    from scintfibrefastsim.response import CellTable, fibre_transport
    from novafastsim_detsim import config, sim
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--fls", type=Path, nargs="+", required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--signs", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="*", default=[])
    ap.add_argument("--max-events", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--monotone", action="store_true",
                    help="collection non-increasing in distance (study F1b, frontend_replay_b_criteria.json)")
    a = ap.parse_args(argv)
    crit = HERE / "frontend_replay_b_criteria.json" if a.monotone else CRIT
    t0 = time.time()
    g = RP.description_arrays(a.description)
    signs = RP.load_signs(a.signs)
    det, params = config.load_description(a.description)
    det, prior = D.apply_fit(det, params, a.params)
    shape = 0.9535
    for f in a.params:
        j = json.loads(f.read_text())
        shape = j.get("detsim", {}).get("dcs_peak_per_charge", shape)
    params["optics.birks_kb"], params["optics.birks_kc"] = 0.0, 0.0      # replay carries EdepBirks
    gg = det.geometry
    P, C = gg.n_planes, gg.n_cells
    lay = layout(det, jnp.float64)
    geo = {"z": np.asarray(lay.z_lo) + 0.5 * gg.cell_depth, "view": np.asarray(lay.view),
           "wc": np.asarray(lay.w_centre), "length": 2.0 * np.asarray(lay.half_len), "readout": gg.readout_end}
    groups_np = np.asarray(sim.apd_groups(det))

    # ---- stock targets (train)
    keep = [i for i, c in enumerate(a.chunk_ids) if c < 10]
    a.chunk_ids = [a.chunk_ids[i] for i in keep]
    a.stock, a.hepevt, a.fls = ([x[i] for i in keep] for x in (a.stock, a.hepevt, a.fls))
    st_train, _ = L2.load(a, "train")
    h_tr, e_tr = L2.stock_hits(st_train, geo, det.settings.apd_group or 32)
    cut = (h_tr[:, 4] < MAX_ADC) & (h_tr[:, 5] < QUIET_MEV)
    R, E, _ = L2.ratio_profile(h_tr[cut], L2.FIT_BINS, len(e_tr), MIN_MEV)
    pe_sat_target = float(np.median(h_tr[h_tr[:, 2] > SAT_MEV, 1]))

    n_all = sum(len(r) for _, r in st_train)
    stride = max(1, int(np.ceil(n_all / a.max_events)))
    work = []                                    # (fls, rows, stock d, global offsets)
    k = 0
    for (d, rows), fpath in zip(st_train, a.fls):
        pick = [r for j, r in enumerate(rows) if (k + j) % stride == 0]
        k += len(rows)
        work.append((dict(np.load(fpath)), np.asarray(pick), d))
    print(f"train events used {sum(len(w[1]) for w in work)} of {n_all}; saturated-hit target {pe_sat_target:.0f} PE",
          flush=True)

    def axis_distance(d, row):
        vtx = np.array([d["ev_vx"][row], d["ev_vy"][row], d["ev_vz"][row]])
        mom = np.array([d["ev_epx"][row], d["ev_epy"][row], d["ev_epz"][row]])
        pl = np.arange(P)
        w = L2.axis_w(vtx, mom, geo["z"], geo["view"])
        return L2.distance(geo, pl, geo["view"], w)                    # [P]

    # ---- optics: linear in the table values (yield folded into them)
    km = KNOTS_CM * 10.0
    delay = np.asarray(fibre_transport(jnp.asarray(km), gg.cell(), config.optics(params))[1])

    def pe_of(gvec, seg):
        p = config.to_device(params, jnp.float64)
        p["optics.yield_per_mev"] = jnp.asarray(1.0)
        p["optics_table"] = CellTable(jnp.asarray(km), gvec, jnp.asarray(delay))
        r = jax.vmap(lambda s: sim.accumulate(det, p, s))(seg)
        return r.pe, r.evis

    jac_pe = jax.jit(jax.jacfwd(lambda gv, s: pe_of(gv, s)[0]))
    evis_only = jax.jit(lambda gv, s: pe_of(gv, s)[1])
    q_max = MAX_ADC / (float(params["apd.adc_per_pe"]) * shape)
    nb = len(L2.FIT_BINS) - 1

    def optics_fit(g_cut):
        W = np.zeros((nb, len(KNOTS_CM)))
        B = np.zeros(nb)
        gv = jnp.asarray(g_cut)
        for fl, rows, d in work:
            for rb, seg in RP.batches(g, fl, rows, signs, "float64", energy="edep_birks"):
                J = np.asarray(jac_pe(gv, seg))                          # [B, P, C, K]
                ev = np.asarray(evis_only(gv, seg))
                pe_now = J @ np.asarray(g_cut)
                for b, row in enumerate(rb):
                    oth = XT.group_others(ev[b], groups_np)
                    m = (ev[b] > MIN_MEV) & (oth < QUIET_MEV) & (pe_now[b] < q_max)
                    dist = np.broadcast_to(axis_distance(d, row)[:, None], (P, C))
                    ib = np.clip(np.digitize(dist[m], L2.FIT_BINS) - 1, 0, nb - 1)
                    np.add.at(W, ib, J[b][m])
                    np.add.at(B, ib, ev[b][m])
        ok = (B > 0) & (E > 0)
        M = W[ok] / (B[ok] * E[ok])[:, None]
        if a.monotone:                      # g = U delta, delta >= 0: cumulative increments from the far end
            U = np.triu(np.ones((len(KNOTS_CM), len(KNOTS_CM))))
            delta, rnorm = nnls(M @ U, R[ok] / E[ok])
            sol = U @ delta
        else:
            sol, rnorm = nnls(M, R[ok] / E[ok])
        return sol, float(rnorm ** 2), int(ok.sum())

    g0 = float(params["optics.yield_per_mev"]) * np.asarray(fibre_transport(jnp.asarray(km), gg.cell(),
                                                                           config.optics(params))[0])
    g1, chi_a, nbins = optics_fit(g0)
    g2, chi_b, _ = optics_fit(g1)
    yield_fit = float(g2[0]) if (a.monotone or g2[0] > 0) else float(np.max(g2))   # monotone: g2[0] is the maximum
    collection = g2 / yield_fit
    params["optics.yield_per_mev"] = yield_fit
    params["optics_table"] = CellTable(km, collection, delay)
    print(f"optics: yield {yield_fit:.2f}, chi2 {chi_a:.1f} -> {chi_b:.1f} over {nbins} bins", flush=True)

    # ---- saturation and crosstalk on the replay
    groups = sim.apd_groups(det)

    @jax.jit
    def respond(p, seg, keys):
        def one(s, k):
            r = sim.accumulate(det, p, s)
            h = nova_apd.sample(k, r.pe, r.time, config.apd(p), det.settings.readout(), groups)
            return r.evis, h[0], h[3]
        return jax.vmap(one)(seg, keys)

    seg_cache = []
    for fl, rows, d in work:
        for rb, seg in RP.batches(g, fl, rows, signs, "float64", energy="edep_birks"):
            keys = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(a.seed), j))(jnp.asarray(rb))
            seg_cache.append((d, rb, seg, keys))

    def readout(over):
        p = dict(params)
        p.update(over)
        pp = config.to_device(p, jnp.float64)
        for d, rb, seg, keys in seg_cache:
            ev, q, hit = (np.asarray(x) for x in respond(pp, seg, keys))
            yield d, rb, ev, q, hit

    # stock efficiency table on the same events (stock deposits = the replay's visible energy)
    stock_tab = np.zeros((2, len(XT.E_BANDS) - 1, len(XT.O_BINS) - 1))
    for d, rb, seg, keys in seg_cache:
        s = (d["hit_true_e"] > 0) & (d["hit_plane"] < P)
        sig_keys = set((d["hit_row"][s].astype(np.int64) * 10 ** 6 + d["hit_plane"][s] * 1000 + d["hit_cell"][s]).tolist())
        for row in rb:
            m = (d["dep_row"] == row) & (d["dep_plane"] < P)
            e = np.zeros((P, C)); hit = np.zeros((P, C), bool)
            e[d["dep_plane"][m], d["dep_cell"][m]] = d["dep_edep_birks"][m] * 1000.0
            hit[d["dep_plane"][m], d["dep_cell"][m]] = [x in sig_keys for x in
                                                        (row * 10 ** 6 + d["dep_plane"][m] * 1000 + d["dep_cell"][m]).tolist()]
            XT.accumulate(stock_tab, e, hit, XT.group_others(e, groups_np))
    es = stock_tab[1] / np.maximum(stock_tab[0], 1)
    vs = es * (1 - es) / np.maximum(stock_tab[0], 1)

    state = {"apd.sat_recovery": float(params["apd.sat_recovery"]), "apd.crosstalk": float(params["apd.crosstalk"])}

    def sat_resid(kappa):
        qs = [q[i][(ev[i] > SAT_MEV) & hit[i]] for d, rb, ev, q, hit in readout({**state, "apd.sat_recovery": kappa})
              for i in range(len(rb))]
        v = float(np.median(np.concatenate(qs)))
        print(f"  sat_recovery {kappa:.4f}: median saturated charge {v:.0f}", flush=True)
        return v - pe_sat_target

    def xt_chi2(kappa):
        tab = np.zeros_like(stock_tab)
        for d, rb, ev, q, hit in readout({**state, "apd.crosstalk": kappa}):
            for i in range(len(rb)):
                XT.accumulate(tab, ev[i], hit[i] & (ev[i] > 0), XT.group_others(ev[i], groups_np))
        ed = tab[1] / np.maximum(tab[0], 1)
        vd = ed * (1 - ed) / np.maximum(tab[0], 1)
        ok = (stock_tab[0] >= 20) & (tab[0] >= 20)
        c = float(np.sum(((ed - es) ** 2 / np.maximum(vs + vd, 1e-12))[ok]))
        print(f"  crosstalk {kappa:.5f}: chi2 {c:.1f} over {int(ok.sum())}", flush=True)
        return c

    rounds = []
    for k in range(1, MAX_ROUNDS + 1):
        prev = dict(state)
        lo, hi = sat_resid(0.0), sat_resid(1.0)
        state["apd.sat_recovery"] = float(brentq(sat_resid, 0.0, 1.0, xtol=1e-3)) if lo * hi < 0 else (0.0 if abs(lo) < abs(hi) else 1.0)
        res = minimize_scalar(xt_chi2, bounds=(0.0, 0.05), method="bounded", options={"xatol": 2e-4})
        state["apd.crosstalk"] = float(res.x)
        rounds.append({"round": k, **state, "crosstalk_chi2": float(res.fun)})
        print(f"round {k}: {state}", flush=True)
        if all(abs(state[n] - prev[n]) <= TOL * max(abs(prev[n]), 1e-12) for n in state):
            break
    out = {
        "criteria_sha256": hashlib.sha256(crit.read_bytes()).hexdigest(), "criteria": crit.name, "monotone": a.monotone,
        "prior_fits": prior, "signs": a.signs.name,
        "events": int(sum(len(w[1]) for w in work)), "seed": a.seed,
        "settings": {"apd_group": det.settings.apd_group},
        "params": {"optics.yield_per_mev": yield_fit, **state},
        "optics_table": {"distance": km.tolist(), "collection": collection.tolist(), "delay": delay.tolist()},
        "optics_fit": {"chi2_pass1": chi_a, "chi2_pass2": chi_b, "bins": nbins, "knots_cm": KNOTS_CM.tolist()},
        "rounds": rounds, "seconds": time.time() - t0,
        "converged": bool(len(rounds) > 1 and all(abs(rounds[-1][n] - rounds[-2][n])
                                                  <= TOL * max(abs(rounds[-2][n]), 1e-12) for n in state)),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("params", "optics_fit", "rounds", "converged", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
