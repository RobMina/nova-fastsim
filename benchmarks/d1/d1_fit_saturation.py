"""D1 front-end saturation: train fits of adc_per_pe, the ADC ceiling and sat_recovery
(benchmarks/d1/saturation_criteria.json).

    python benchmarks/d1/d1_fit_saturation.py OUT.json --description ND.json --params DEPOSIT.json STOCHASTIC.json
        --stock CHUNK.npz ... --hepevt CHUNK.txt ... --chunk-ids ID ... [--max-events 1500] [--seed 13]

Only train chunks (ids 0-9) are accepted. OUT.json holds a ``params`` override dict (``apd.adc_per_pe``,
``apd.adc_max``, ``apd.adc_max_sigma``, ``apd.sat_recovery``) for ``d1_l2.py --params``, applied after the earlier
fits; the L2 run must use ``--fit-max-adc 3000``. Stock data and results are collaboration data: write OUT.json to a
private area.
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
import d1_l2 as L2  # noqa: E402

CRIT = HERE / "saturation_criteria.json"
FIT_MAX_ADC = 3000.0
ADC_LO, ADC_HI, E_SAT = 100.0, 3000.0, 100.0


def detsim_signal_hits(det, params, parts, seed, batch=50):
    """Signal hits (above threshold, visible energy > 0) of one sampled readout per event: charge, ADC, visible MeV."""
    import jax
    import jax.numpy as jnp
    from novafastsim_detsim import config, sim
    p = config.to_device(params, jnp.float64)
    key = jax.random.PRNGKey(seed)
    n = np.asarray(parts.pdg).shape[0]
    out = []
    for i in range(0, n, batch):
        sl = np.arange(i, min(i + batch, n))
        sub = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[sl], jnp.float64)
                                     if np.issubdtype(np.asarray(a).dtype, np.floating) else jnp.asarray(np.asarray(a)[sl]), parts)
        r, h = sim.sampled_batch(det, p, sub, jax.random.fold_in(key, i))
        sig = np.asarray(h.hit) & (np.asarray(r.evis) > 0)
        out.append(np.c_[np.asarray(h.pe, np.float64)[sig], np.asarray(h.adc, np.float64)[sig],
                         np.asarray(r.evis, np.float64)[sig]])
    return np.concatenate(out)


def main(argv=None) -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from scipy.optimize import brentq
    from novafastsim_detsim import config
    from novafastsim_detsim.geometry import layout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="*", default=[], help="earlier fit outputs, in order")
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    ap.add_argument("--max-events", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fit-quiet-mev", type=float,
                    help="optics fit only on stock hits whose board's other cells hold less than this Birks MeV")
    a = ap.parse_args(argv)
    if not (len(a.stock) == len(a.hepevt) == len(a.chunk_ids)):
        raise SystemExit("--stock, --hepevt and --chunk-ids need the same length")
    if any(c >= 10 for c in a.chunk_ids):
        raise SystemExit("the saturation fits use train chunks (ids 0-9) only")
    t0 = time.time()

    det, params = config.load_description(a.description)
    det, prior = D.apply_fit(det, params, a.params)
    g = det.geometry
    lay = layout(det, jax.numpy.float64)
    geo = {"z": np.asarray(lay.z_lo) + 0.5 * g.cell_depth, "view": np.asarray(lay.view),
           "wc": np.asarray(lay.w_centre), "length": g.cell_length, "readout": g.readout_end}

    # ---- optics (level-2 fit with the ADC cut), as d1_l2.py --fit-max-adc does
    st_train, parts = L2.load(a, "train")
    h_tr, e_tr = L2.stock_hits(st_train, geo, det.settings.apd_group or 32)
    h_fit = h_tr[h_tr[:, 4] < FIT_MAX_ADC]
    if a.fit_quiet_mev is not None:
        h_fit = h_fit[h_fit[:, 5] < a.fit_quiet_mev]
    R, E, _ = L2.ratio_profile(h_fit, L2.FIT_BINS, len(e_tr), L2.FIT_MIN_MEV)
    centres = 0.5 * (L2.FIT_BINS[1:] + L2.FIT_BINS[:-1])
    optics = L2.fit_optics(det, params, R, E, centres)
    params.update({k: v for k, v in optics.items() if k.startswith("optics.")})

    # ---- stock targets
    pe_s, mev_s, adc_s = h_tr[:, 1], h_tr[:, 2], h_tr[:, 4]
    lin = (adc_s > ADC_LO) & (adc_s < ADC_HI)
    adc_per_pe_stock = float(np.median(adc_s[lin] / pe_s[lin]))
    sat = mev_s > E_SAT
    q16, q50, q84 = np.quantile(adc_s[sat], [0.16, 0.5, 0.84])
    adc_max, adc_max_sigma = float(q50), float(0.5 * (q84 - q16))
    pe_sat_target = float(np.median(pe_s[sat]))
    print(f"stock: ADC/PE {adc_per_pe_stock:.4f}; ceiling {adc_max:.0f} +- {adc_max_sigma:.0f}; saturated-hit PE "
          f"median {pe_sat_target:.0f} ({int(sat.sum())} hits)", flush=True)

    n_all = np.asarray(parts.pdg).shape[0]
    take = np.arange(n_all)[::max(1, int(np.ceil(n_all / a.max_events)))]
    sub = jax.tree_util.tree_map(lambda x: np.asarray(x)[take], parts)

    # ---- adc_per_pe: detsim's DCS peak per unit charge with adc_per_pe = 1 and no ceiling
    p1 = dict(params)
    p1.update({"apd.adc_per_pe": 1.0, "apd.adc_max": float("inf"), "apd.adc_max_sigma": 0.0, "apd.sat_recovery": 0.0})
    h1 = detsim_signal_hits(det, p1, sub, a.seed)
    lin_d = (h1[:, 1] > ADC_LO / adc_per_pe_stock) & (h1[:, 1] < ADC_HI / adc_per_pe_stock)
    shape = float(np.median(h1[lin_d, 1] / h1[lin_d, 0]))
    adc_per_pe = adc_per_pe_stock / shape
    print(f"detsim DCS peak per unit charge {shape:.4f} -> adc_per_pe {adc_per_pe:.4f}", flush=True)

    # ---- sat_recovery
    ps = dict(params)
    ps.update({"apd.adc_per_pe": adc_per_pe, "apd.adc_max": adc_max, "apd.adc_max_sigma": adc_max_sigma})

    def med_sat(k):
        ps["apd.sat_recovery"] = k
        h = detsim_signal_hits(det, ps, sub, a.seed)
        s = h[:, 2] > E_SAT
        v = float(np.median(h[s, 0]))
        print(f"  sat_recovery {k:.4f}: median charge of saturated hits {v:.1f} ({int(s.sum())} hits)", flush=True)
        return v - pe_sat_target, h

    f_lo, h_lo = med_sat(0.0)
    f_hi, _ = med_sat(1.0)
    if f_lo * f_hi < 0:
        k = brentq(lambda x: med_sat(x)[0], 0.0, 1.0, xtol=1e-3)
        note = "root"
    else:
        k, note = (0.0, "at lower bound") if abs(f_lo) < abs(f_hi) else (1.0, "at upper bound")
    _, h_fin = med_sat(k)
    adc_fin = h_fin[h_fin[:, 2] > E_SAT, 1]

    out = {
        "criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(),
        "prior_fits": prior, "chunks": a.chunk_ids, "events": int(len(take)), "seed": a.seed,
        "optics_used": optics, "fit_max_adc": FIT_MAX_ADC, "fit_quiet_mev": a.fit_quiet_mev,
        "params": {"apd.adc_per_pe": adc_per_pe, "apd.adc_max": adc_max, "apd.adc_max_sigma": adc_max_sigma,
                   "apd.sat_recovery": float(k)},
        "stock": {"adc_per_pe": adc_per_pe_stock, "saturated_hits": int(sat.sum()), "saturated_pe_median": pe_sat_target,
                  "saturated_adc_q16_50_84": [float(q16), float(q50), float(q84)]},
        "detsim": {"dcs_peak_per_charge": shape, "saturated_adc_q16_50_84": np.quantile(adc_fin, [0.16, 0.5, 0.84]).tolist()},
        "sat_recovery_note": note, "seconds": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
