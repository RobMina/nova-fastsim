"""D1 muon arm, check C1 (benchmarks/d1/muon_checks_criteria.json): the muon-catcher planes' local-to-world axis signs
and the plane order, from stock muon deposits.

    python benchmarks/d1/d1_muon_frame.py OUT.json --description ND_MC.json --signs SIGNS.json
        --fls CHUNK_fls.npz ... --chunk-ids ID ...

``--signs`` holds the main planes' signs (the electron replay's). OUT.json holds them plus ``signs_mc`` for the
catcher planes, for ``d1_replay.py --signs``. Train chunks (ids 0-9) fit the signs; test chunks (10-19) give the
plane-order fraction. Stock data and results are collaboration data: keep OUT.json private.
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
import d1_replay as RP  # noqa: E402

CRIT = HERE / "muon_checks_criteria.json"
COMBOS = list(itertools.product((1, -1), repeat=3))


def muon_pairs(g, f, signs):
    """Per event, time-ordered muon deposit midpoints (world, with ``signs``) -> consecutive pairs with at least one
    catcher deposit: perpendicular step (cm, against the muon's initial direction) and z steps."""
    rows = np.flatnonzero(D.selected({k: f[k] for k in f if k.startswith("ev_")}))
    keep = np.isin(f["fls_row"], rows) & (np.abs(f["fls_pdg"]) == 13) & (f["fls_plane"] < len(g["z"]))
    idx = np.flatnonzero(keep)
    pl, ce = f["fls_plane"][idx], f["fls_cell"][idx]
    mid = 0.5 * (RP.to_world(g, pl, ce, f["fls_x0"][idx], f["fls_y0"][idx], f["fls_z0"][idx], signs)
                 + RP.to_world(g, pl, ce, f["fls_x1"][idx], f["fls_y1"][idx], f["fls_z1"][idx], signs))
    r, t, cat = f["fls_row"][idx], f["fls_t"][idx], g["catcher"][pl]
    order = np.lexsort((t, r))
    r, mid, cat = r[order], mid[order], cat[order]
    same = r[1:] == r[:-1]
    any_cat = (cat[1:] | cat[:-1]) & same
    d = np.c_[f["ev_epx"][r], f["ev_epy"][r], f["ev_epz"][r]][1:]
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    step = mid[1:] - mid[:-1]
    perp = np.linalg.norm(step - (step * d).sum(1, keepdims=True) * d, axis=1)
    return perp[any_cat], step[same, 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--signs", type=Path, required=True)
    ap.add_argument("--fls", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    g = RP.description_arrays(a.description)
    main_signs = RP.load_signs(a.signs)
    train = [dict(np.load(p)) for c, p in zip(a.chunk_ids, a.fls) if c < 10]
    test = [dict(np.load(p)) for c, p in zip(a.chunk_ids, a.fls) if c >= 10]
    table = {}
    for c0, c1 in itertools.product(COMBOS, COMBOS):
        s = dict(main_signs)
        s["mc"] = {0: c0, 1: c1}
        perp = np.concatenate([muon_pairs(g, f, s)[0] for f in train])
        table[f"{c0}|{c1}"] = float(np.median(perp))
    best = min(table, key=table.get)
    b0, b1 = (tuple(int(x) for x in part.strip("()").split(",")) for part in best.split("|"))

    def flips_u_or_w(key):
        c0, c1 = (tuple(int(x) for x in part.strip("()").split(",")) for part in key.split("|"))
        return any(c[0] != b[0] or c[2] != b[2] for c, b in ((c0, b0), (c1, b1)))

    rival = min(v for k, v in table.items() if flips_u_or_w(k))
    s = dict(main_signs)
    s["mc"] = {0: b0, 1: b1}
    dz = np.concatenate([muon_pairs(g, f, s)[1] for f in test])
    order_frac = float(np.mean(dz >= -1.0))
    out = {"criteria_sha256": hashlib.sha256(CRIT.read_bytes()).hexdigest(),
           "signs": {str(k): list(v) for k, v in main_signs.items() if k != "mc"},
           "signs_mc": {"0": list(b0), "1": list(b1)},
           "best_median_perp_cm": table[best], "best_rival_flipping_u_or_w_cm": rival,
           "plane_order_fraction": order_frac, "test_pairs": int(len(dz)),
           "pass": bool(table[best] < 5.0 and rival >= 2.0 * table[best] and order_frac > 0.98),
           "table_median_perp_cm": table}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "table_median_perp_cm"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
