"""Single-particle HEPEVT chunks for the D1 muon/hadron arms (benchmarks/d1/guns.json).

    python benchmarks/d1/make_guns.py OUT_DIR [--spec guns.json]

Writes OUT_DIR/gun_<NNNN>.txt (flat chunk index over all species, in the spec's order) and OUT_DIR/manifest.txt
(index, species, split, seed, sha256). HEPEVT lines: ``status pdg m1 m2 d1 d2 px py pz E m x y z t`` in GeV, cm and
ns, one status-1 particle per event, events numbered from 1 in each chunk. Synthetic: no collaboration input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MASS_GEV = {13: 0.1056583755, 211: 0.13957039, 2212: 0.93827208816, 2112: 0.93956542052, 111: 0.1349768,
            22: 0.0, 11: 0.00051099895}


def chunk_text(rng, sp, spec):
    n = spec["events_per_chunk"]
    box, ct, tw = spec["vertex_box_cm"], spec["cos_theta"], spec["time_ns"]
    pdgs = np.asarray(sp["pdg"])
    pdg = pdgs[rng.integers(0, len(pdgs), n)]
    lo, hi = np.log(sp["t_kin_gev"][0]), np.log(sp["t_kin_gev"][1])
    tk = np.exp(rng.uniform(lo, hi, n))
    m = np.array([MASS_GEV[abs(int(q))] for q in pdg])
    e = tk + m
    p = np.sqrt(np.maximum(e * e - m * m, 0.0))
    c = rng.uniform(ct[0], ct[1], n)
    s = np.sqrt(1.0 - c * c)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    px, py, pz = p * s * np.cos(phi), p * s * np.sin(phi), p * c
    x, y, z = (rng.uniform(*box[k], n) for k in ("x", "y", "z"))
    t = rng.uniform(tw[0], tw[1], n)
    lines = []
    for i in range(n):
        lines.append(f"{i + 1} 1")
        lines.append(f"1 {int(pdg[i])} 0 0 0 0 {px[i]:.8g} {py[i]:.8g} {pz[i]:.8g} {e[i]:.8g} {m[i]:.8g} "
                     f"{x[i]:.6g} {y[i]:.6g} {z[i]:.6g} {t[i]:.7g}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--spec", type=Path, default=HERE / "guns.json")
    a = ap.parse_args(argv)
    spec = json.loads(a.spec.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    rows = ["# index species split seed sha256  (spec sha256 %s)" % hashlib.sha256(a.spec.read_bytes()).hexdigest()]
    idx = 0
    for sp in spec["species"]:
        for k in range(sp["chunks"]):
            seed = spec["seed"] * 1000 + idx
            txt = chunk_text(np.random.default_rng(seed), sp, spec)
            f = a.out / f"gun_{idx:04d}.txt"
            f.write_text(txt)
            split = "train" if k < sp["chunks"] // 2 else "test"
            rows.append(f"{idx:04d} {sp['name']} {split} {seed} {hashlib.sha256(txt.encode()).hexdigest()}")
            idx += 1
    (a.out / "manifest.txt").write_text("\n".join(rows) + "\n")
    print(f"{idx} chunks in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
