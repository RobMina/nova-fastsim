"""D1 joint electron + muon light-curve fit on replayed stock deposits (benchmarks/d1/joint_light_criteria.json).

    python benchmarks/d1/d1_joint_light.py OUT.json --description ND_MC.json --signs SIGNS.json --params FIT.json ...
        --e-fls ... --e-stock ... --e-ids ... --mu-fls ... --mu-stock ... --mu-ids ... [--max-events 1000]

Train chunks (ids 0-9) only. OUT.json holds ``params`` (``optics.yield_per_mev``) and ``optics_table`` for
``d1_replay_frontend.py --optics-from`` (saturation and crosstalk) and the level-2 runs. Stock data and results are
collaboration data: keep OUT.json private.
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
import d1_replay as RP  # noqa: E402

CRIT = HERE / "joint_light_criteria.json"
KNOTS_CM = np.r_[0.0, 10.0, np.arange(20.0, 400.0 + 1e-9, 20.0)]
BINS = np.arange(0.0, 400.0 + 1e-9, 10.0)
MIN_MEV, QUIET_MEV, MAX_ADC = 2.0, 20.0, 3000.0


def transform():
    """g = T theta: theta = [g_0cm, g_10cm, increments from 20 cm on] >= 0, non-increasing from 20 cm."""
    K = len(KNOTS_CM)
    T = np.zeros((K, K))
    T[0, 0] = T[1, 1] = 1.0
    T[2:, 2:] = np.triu(np.ones((K - 2, K - 2)))
    return T


def cell_distance(g, f, row, lengths, readout, signs, P, C):
    """[P, C] distance to the readout from the energy-weighted long-axis position of the cell's own deposits (nan
    where the event left no deposit)."""
    s = (f["fls_row"] == row) & (f["fls_plane"] < P)
    pl, ce, e = f["fls_plane"][s], f["fls_cell"][s], f["fls_edep_birks"][s]
    zl = 0.5 * (f["fls_z0"][s] + f["fls_z1"][s])
    view = g["view"][pl]
    sw = np.where(g["catcher"][pl], np.array([signs["mc"][0][2], signs["mc"][1][2]])[view],
                  np.array([signs[0][2], signs[1][2]])[view])
    num, den = np.zeros((P, C)), np.zeros((P, C))
    np.add.at(num, (pl, ce), e * sw * zl)
    np.add.at(den, (pl, ce), e)
    wrel = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    end = np.asarray(readout)[g["view"][:P]][:, None]
    return np.clip(0.5 * lengths[:P, None] - end * wrel, 0.0, lengths[:P, None]), den


def sample_events(fls_paths, stock_paths, ids, max_events):
    work = []
    for c, fp, sp in zip(ids, fls_paths, stock_paths):
        if c >= 10:
            continue
        d = dict(np.load(sp))
        work.append((dict(np.load(fp)), np.flatnonzero(D.selected(d)), d))
    n_all = sum(len(r) for _, r, _ in work)
    stride = max(1, int(np.ceil(n_all / max_events)))
    out, k = [], 0
    for f, rows, d in work:
        pick = np.array([r for j, r in enumerate(rows) if (k + j) % stride == 0], int)
        k += len(rows)
        out.append((f, pick, d))
    return out, n_all


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import nnls
    from scintfibrefastsim.response import CellTable, fibre_transport
    from novafastsim_detsim import config, sim

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--signs", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="*", default=[])
    for s in ("e", "mu"):
        ap.add_argument(f"--{s}-fls", type=Path, nargs="+", required=True)
        ap.add_argument(f"--{s}-stock", type=Path, nargs="+", required=True)
        ap.add_argument(f"--{s}-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-events", type=int, default=1000)
    a = ap.parse_args(argv)
    t0 = time.time()
    g = RP.description_arrays(a.description)
    desc = json.loads(a.description.read_text())
    signs = RP.load_signs(a.signs)
    det, params = config.load_description(a.description)
    det, prior = D.apply_fit(det, params, a.params)
    shape = 0.9535
    for f in a.params:
        shape = json.loads(f.read_text()).get("detsim", {}).get("dcs_peak_per_charge", shape)
    params["optics.birks_kb"], params["optics.birks_kc"] = 0.0, 0.0
    gg = det.geometry
    P, C = gg.n_planes, gg.n_cells
    lengths = np.asarray(gg.lengths())
    readout = desc.get("readout_end", [1, 1])
    groups_np = np.asarray(sim.apd_groups(det))
    km = KNOTS_CM * 10.0
    delay = np.asarray(fibre_transport(jnp.asarray(km), gg.cell(), config.optics(params))[1])
    q_max = MAX_ADC / (float(params["apd.adc_per_pe"]) * shape)
    nb = len(BINS) - 1

    def pe_of(gvec, seg):
        p = config.to_device(params, jnp.float64)
        p["optics.yield_per_mev"] = jnp.asarray(1.0)
        p["optics_table"] = CellTable(jnp.asarray(km), gvec, jnp.asarray(delay))
        r = jax.vmap(lambda s_: sim.accumulate(det, p, s_))(seg)
        return r.pe, r.evis

    jac_pe = jax.jit(jax.jacfwd(lambda gv, s_: pe_of(gv, s_)[0]))
    evis_only = jax.jit(lambda gv, s_: pe_of(gv, s_)[1])

    # starting curve: the prior fits' table (F1b) evaluated at the new knots
    t_prev = params.get("optics_table")
    y_prev = float(params["optics.yield_per_mev"])
    g_start = y_prev * (np.interp(km, np.asarray(t_prev.distance), np.asarray(t_prev.collection)) if t_prev is not None
                        else np.asarray(fibre_transport(jnp.asarray(km), gg.cell(), config.optics(params))[0]))

    samples = {}
    for name in ("e", "mu"):
        work, n_all = sample_events(getattr(a, f"{name}_fls"), getattr(a, f"{name}_stock"), getattr(a, f"{name}_ids"),
                                    a.max_events)
        # stock target per event (delta-method errors)
        A_ev, B_ev = [], []
        for f, rows, d in work:
            for row in rows:
                dist, den = cell_distance(g, f, row, lengths, readout, signs, P, C)
                oth = XT.group_others(den * 1000.0, groups_np)
                s = (d["hit_row"] == row) & (d["hit_plane"] < P) & (d["hit_true_e"] > 0)
                hp, hc = d["hit_plane"][s], d["hit_cell"][s]
                mev = d["hit_true_e_birks"][s] * 1000.0
                dd = dist[hp, hc]
                m = (mev > MIN_MEV) & (d["hit_adc"][s] < MAX_ADC) & (oth[hp, hc] < QUIET_MEV) & np.isfinite(dd)
                ib = np.clip(np.digitize(dd[m], BINS) - 1, 0, nb - 1)
                Ae, Be = np.zeros(nb), np.zeros(nb)
                np.add.at(Ae, ib, d["hit_pe"][s][m])
                np.add.at(Be, ib, mev[m])
                A_ev.append(Ae)
                B_ev.append(Be)
        A_ev, B_ev = np.array(A_ev), np.array(B_ev)
        sa, sb = A_ev.sum(0), B_ev.sum(0)
        R = np.where(sb > 0, sa / np.maximum(sb, 1e-30), 0.0)
        E = np.where(sb > 0, np.sqrt(((A_ev - R * B_ev) ** 2).sum(0)) / np.maximum(sb, 1e-30), 0.0)
        samples[name] = {"work": work, "R": R, "E": E, "events": int(len(A_ev)), "events_available": n_all}
        print(f"{name}: {len(A_ev)} events of {n_all}; stock PE/MeV first bins {np.round(R[:4], 2).tolist()}", flush=True)

    def system(g_cut):
        rows_M, rows_t, per = [], [], {}
        gv = jnp.asarray(g_cut)
        for name, smp in samples.items():
            W, B = np.zeros((nb, len(KNOTS_CM))), np.zeros(nb)
            for f, rows, d in smp["work"]:
                for rb, seg in RP.batches(g, f, rows, signs, "float64", energy="edep_birks"):
                    J = np.asarray(jac_pe(gv, seg))
                    ev = np.asarray(evis_only(gv, seg))
                    pe_now = J @ np.asarray(g_cut)
                    for b, row in enumerate(rb):
                        dist, _ = cell_distance(g, f, row, lengths, readout, signs, P, C)
                        oth = XT.group_others(ev[b], groups_np)
                        m = (ev[b] > MIN_MEV) & (oth < QUIET_MEV) & (pe_now[b] < q_max) & np.isfinite(dist)
                        ib = np.clip(np.digitize(dist[m], BINS) - 1, 0, nb - 1)
                        np.add.at(W, ib, J[b][m])
                        np.add.at(B, ib, ev[b][m])
            ok = (B > 0) & (smp["E"] > 0)
            rows_M.append(W[ok] / (B[ok] * smp["E"][ok])[:, None])
            rows_t.append(smp["R"][ok] / smp["E"][ok])
            per[name] = (W, B, ok)
        return np.vstack(rows_M), np.concatenate(rows_t), per

    T = transform()
    fits = []
    g_cut = g_start
    for _ in range(2):
        M, tgt, per = system(g_cut)
        theta, rnorm = nnls(M @ T, tgt)
        g_new = T @ theta
        chi = {}
        for name, (W, B, ok) in per.items():
            mod = (W @ g_new) / np.maximum(B, 1e-30)
            r = ((mod - samples[name]["R"]) / np.maximum(samples[name]["E"], 1e-30))[ok]
            chi[name] = {"chi2": float(np.sum(r ** 2)), "bins": int(ok.sum()),
                         "model_first_bins": np.round(mod[:4], 2).tolist()}
        fits.append(chi)
        print(f"pass: {chi}", flush=True)
        g_cut = g_new
    yield_fit = float(g_cut[2])
    collection = g_cut / yield_fit
    out = {"criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(), "criteria": CRIT.name,
           "prior_fits": prior, "signs": a.signs.name,
           "params": {"optics.yield_per_mev": yield_fit},
           "optics_table": {"distance": km.tolist(), "collection": collection.tolist(), "delay": delay.tolist()},
           "optics_fit": {"passes": fits, "bins": int(sum(v["bins"] for v in fits[-1].values())),
                          "knots_cm": KNOTS_CM.tolist(), "events": {k: v["events"] for k, v in samples.items()}},
           "stock_profiles": {k: {"bins_cm": BINS.tolist(), "pe_per_mev": v["R"].tolist(), "errors": v["E"].tolist()}
                              for k, v in samples.items()},
           "seconds": time.time() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("params", "optics_fit", "seconds")}, indent=1))
    print("collection", np.round(collection, 3).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
