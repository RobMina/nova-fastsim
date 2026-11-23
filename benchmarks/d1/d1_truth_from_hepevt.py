"""Fill a converted stock file's primary-particle fields from its HepEVT chunk, for samples whose primary is not an
electron (D1HitDump records the true primary only for |pdg| 11):

    python benchmarks/d1/d1_truth_from_hepevt.py IN.npz CHUNK.txt OUT.npz

Works on both converters' outputs (``*_stock.npz``, ``*_fls.npz``): stock event k is HepEVT event ``ev_evt[k]``, whose
first status-1 particle is the generator's primary. ``ev_eE`` and ``ev_ep*`` in GeV, ``ev_v*`` in cm, ``ev_vt`` in ns,
as the dump writes them; ``ev_pdg`` is added and ``truth_source`` records the patch. Collaboration data in and out:
keep the output private.
"""
from __future__ import annotations

import sys

import numpy as np


def main(inp, hepevt, out) -> int:
    from novafastsim_detsim.particles import MASS, from_hepevt
    d = dict(np.load(inp))
    p = from_hepevt(open(hepevt).read())
    k = d["ev_evt"].astype(int) - 1
    pdg = np.asarray(p.pdg)[k, 0]
    mom = np.asarray(p.momentum)[k, 0] / 1000.0
    m = np.array([MASS.get(abs(int(q)), 0.0) for q in pdg]) / 1000.0
    vtx = np.asarray(p.vertex)[k, 0]
    d.update(ev_eE=np.sqrt((mom ** 2).sum(1) + m ** 2), ev_epx=mom[:, 0], ev_epy=mom[:, 1], ev_epz=mom[:, 2],
             ev_vx=vtx[:, 0], ev_vy=vtx[:, 1], ev_vz=vtx[:, 2], ev_vt=np.asarray(p.time)[k, 0], ev_pdg=pdg,
             truth_source=np.array("hepevt"))
    np.savez_compressed(out, **d)
    print(out, "events", len(k), "pdg", sorted({int(q) for q in pdg}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
