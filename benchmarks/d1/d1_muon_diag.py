"""D1 muon arm, post-hoc diagnostics after study MU0 (labelled; not a scored check):

    python benchmarks/d1/d1_muon_diag.py OUT.json --description ND_MC.json --signs SIGNS_MC.json
        --fls CHUNK_fls.npz ... --stock CHUNK_stock.npz ... --chunk-ids ID ...

1. Plane order against the through-depth sign: the fraction of time-consecutive muon deposit pairs whose world z does
   not decrease by more than 1 cm, for every choice of s_z in the main and catcher views (the other signs as given).
2. Readout end of the catcher planes: stock signal-hit PE per true Birks MeV against the distance to the readout
   computed from the cell's own deposits (energy-weighted long-axis position), separately for main and catcher planes
   and per view, for readout at +w and at -w.

Test chunks (ids 10-19) only. Stock data and results are collaboration data: keep OUT.json private.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_compare as D  # noqa: E402
import d1_muon_frame as MF  # noqa: E402
import d1_replay as RP  # noqa: E402

BINS = np.arange(0.0, 400.0 + 1e-9, 20.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--signs", type=Path, required=True)
    ap.add_argument("--fls", type=Path, nargs="+", required=True)
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    g = RP.description_arrays(a.description)
    desc = json.loads(a.description.read_text())
    lengths = np.array([p.get("length", desc["constants"]["cell_length"]) for p in desc["planes"]])
    base = RP.load_signs(a.signs)
    pairs = [(dict(np.load(f)), dict(np.load(s))) for c, f, s in zip(a.chunk_ids, a.fls, a.stock) if c >= 10]

    # 1. plane order vs s_z (main views x catcher views)
    order = {}
    for z0, z1, m0, m1 in itertools.product((1, -1), repeat=4):
        s = {0: (base[0][0], z0, base[0][2]), 1: (base[1][0], z1, base[1][2]),
             "mc": {0: (base["mc"][0][0], m0, base["mc"][0][2]), 1: (base["mc"][1][0], m1, base["mc"][1][2])}}
        dz = np.concatenate([MF.muon_pairs(g, f, s)[1] for f, _ in pairs])
        order[f"main_sz=({z0},{z1}) catcher_sz=({m0},{m1})"] = float(np.mean(dz >= -1.0))

    # 2. readout end: PE/MeV vs deposit-based distance, main and catcher, per view, readout +w / -w
    prof = {}
    acc = {}
    for f, d in pairs:
        rows = np.flatnonzero(D.selected({k: d[k] for k in d if k.startswith("ev_")}))
        keep = np.isin(f["fls_row"], rows) & (f["fls_plane"] < len(g["z"]))
        r, pl, ce = f["fls_row"][keep], f["fls_plane"][keep], f["fls_cell"][keep]
        e = f["fls_edep_birks"][keep]
        zl = 0.5 * (f["fls_z0"][keep] + f["fls_z1"][keep])
        sw = np.where(g["catcher"][pl], np.array([base["mc"][0][2], base["mc"][1][2]])[g["view"][pl]],
                      np.array([base[0][2], base[1][2]])[g["view"][pl]])
        key = (r.astype(np.int64) * 1000 + pl) * 1000 + ce
        u, inv = np.unique(key, return_inverse=True)
        wsum = np.bincount(inv, e * sw * zl)
        esum = np.bincount(inv, e)
        wrel = np.where(esum > 0, wsum / np.maximum(esum, 1e-12), 0.0)          # long-axis offset from the centre
        hk = (d["hit_row"].astype(np.int64) * 1000 + d["hit_plane"]) * 1000 + d["hit_cell"]
        sig = (d["hit_true_e"] > 0) & np.isin(hk, u)
        j = np.searchsorted(u, hk[sig])
        hp = d["hit_plane"][sig]
        for end in (1, -1):
            dist = np.clip(0.5 * lengths[hp] - end * wrel[j], 0.0, lengths[hp])
            for region, rm in (("main", ~g["catcher"][hp]), ("catcher", g["catcher"][hp])):
                for vw in (0, 1):
                    m = rm & (g["view"][hp] == vw) & (d["hit_true_e_birks"][sig] * 1000.0 > 2.0)
                    b = np.clip(np.digitize(dist[m], BINS) - 1, 0, len(BINS) - 2)
                    k = (region, vw, end)
                    A, B = acc.get(k, (np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)))
                    np.add.at(A, b, d["hit_pe"][sig][m])
                    np.add.at(B, b, d["hit_true_e_birks"][sig][m] * 1000.0)
                    acc[k] = (A, B)
    for (region, vw, end), (A, B) in acc.items():
        prof[f"{region} view {vw} readout {'+w' if end > 0 else '-w'}"] = {
            "pe_per_mev": np.round(np.where(B > 0, A / np.maximum(B, 1e-12), 0.0), 2).tolist(),
            "mev": np.round(B, 0).tolist()}
    out = {"label": "post-hoc diagnostic after study MU0, not a scored check", "bins_cm": BINS.tolist(),
           "plane_order_fraction": order, "pe_per_mev_vs_deposit_distance": prof}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"plane_order_fraction": order}, indent=1))
    for k, v in prof.items():
        print(k, v["pe_per_mev"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
