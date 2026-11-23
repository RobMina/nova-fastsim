"""Stock per-deposit tables from D1HitDump trees (``d1dump/hits`` in the fullchain hist files), with uproot:

    python benchmarks/d1/d1_fls_to_npz.py OUT.npz HIST.root [HIST.root ...]

Per event: ``ev_evt`` (HepEVT event number) and the true electron. Per FLS hit (one per particle per cell): event
row, plane, cell, pdg, Edep and EdepBirks (GeV), midpoint time (ns), true path length (cm), and the entry and exit
points in the cell's local frame (cm). Collaboration data: keep the output in a private area.
"""
from __future__ import annotations

import sys

import numpy as np

EV = ("evt", "eE", "epx", "epy", "epz", "vx", "vy", "vz", "vt")
FLS = {"plane": "fPlane", "cell": "fCell", "pdg": "fPDG", "edep": "fEdep", "edep_birks": "fEdepBirks", "t": "fT",
       "len": "fLen", "x0": "fX0", "y0": "fY0", "z0": "fZ0", "x1": "fX1", "y1": "fY1", "z1": "fZ1"}


def main(out, files) -> int:
    import awkward as ak
    import uproot
    ev = {k: [] for k in EV}
    fl = {k: [] for k in FLS}
    rows = []
    n0 = 0
    for fn in files:
        a = uproot.open(fn)["d1dump/hits"].arrays(list(EV) + list(FLS.values()))
        for k in EV:
            ev[k].append(ak.to_numpy(a[k]))
        counts = ak.to_numpy(ak.num(a["fPlane"]))
        rows.append(np.repeat(np.arange(n0, n0 + len(a)), counts))
        for k, b in FLS.items():
            fl[k].append(ak.to_numpy(ak.flatten(a[b])))
        n0 += len(a)
    d = {f"ev_{k}": np.concatenate(v) for k, v in ev.items()}
    d["fls_row"] = np.concatenate(rows).astype(np.int32)
    d.update({f"fls_{k}": np.concatenate(v) for k, v in fl.items()})
    np.savez_compressed(out, **d)
    print(out, "events", n0, "deposits", len(d["fls_row"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
