"""D1 consistent front-end chain (benchmarks/d1/chain_criteria.json): alternate the saturation and crosstalk train fits
with the quiet-board optics until both settle.

    python benchmarks/d1/d1_fit_chain.py OUT_DIR --description ND.json --params DEPOSIT.json STOCHASTIC.json
        --start-crosstalk X1_CROSSTALK.json --stock CHUNK.npz ... --hepevt CHUNK.txt ... --chunk-ids ID ...

Writes OUT_DIR/saturation_r<k>.json, OUT_DIR/crosstalk_r<k>.json and OUT_DIR/chain.json (criteria sha256, every fit
file's sha256, the per-round values, the convergence state and the final pair to pass to d1_l2.py). Only train chunks
(ids 0-9) are accepted by the two fits. Collaboration data: point OUT_DIR at a private area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_fit_crosstalk  # noqa: E402
import d1_fit_saturation  # noqa: E402

CRIT = HERE / "chain_criteria.json"
QUIET_MEV, MAX_ROUNDS, TOL = 20.0, 4, 0.01


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--description", type=Path, required=True)
    ap.add_argument("--params", type=Path, nargs="+", required=True, help="deposit and stochastic fits")
    ap.add_argument("--start-crosstalk", type=Path, required=True, help="the crosstalk fit that starts round 1")
    ap.add_argument("--stock", type=Path, nargs="+", required=True)
    ap.add_argument("--hepevt", type=Path, nargs="+", required=True)
    ap.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    common = (["--description", str(a.description), "--stock", *map(str, a.stock), "--hepevt", *map(str, a.hepevt),
               "--chunk-ids", *map(str, a.chunk_ids)])
    xt = a.start_crosstalk
    prev = None
    rounds = []
    converged = False
    for k in range(1, MAX_ROUNDS + 1):
        sat = a.out / f"saturation_r{k}.json"
        d1_fit_saturation.main([str(sat), *common, "--params", *map(str, a.params), str(xt),
                                "--fit-quiet-mev", str(QUIET_MEV)])
        xt = a.out / f"crosstalk_r{k}.json"
        d1_fit_crosstalk.main([str(xt), *common, "--params", *map(str, a.params), str(sat)])
        s, x = json.loads(sat.read_text()), json.loads(xt.read_text())
        cur = {"sat_recovery": s["params"]["apd.sat_recovery"], "crosstalk": x["params"]["apd.crosstalk"],
               "yield_per_mev": s["optics_used"]["optics.yield_per_mev"], "adc_per_pe": s["params"]["apd.adc_per_pe"],
               "crosstalk_chi2": x["chi2"]}
        rounds.append({"round": k, "saturation": sat.name, "saturation_sha256": sha(sat), "crosstalk": xt.name,
                       "crosstalk_sha256": sha(xt), **cur})
        print(f"chain round {k}: {cur}", flush=True)
        if prev is not None and all(abs(cur[n] - prev[n]) <= TOL * max(abs(prev[n]), 1e-12)
                                    for n in ("sat_recovery", "crosstalk")):
            converged = True
            break
        prev = cur
    rec = {"criteria_sha256": sha(CRIT), "inputs": {str(p.name): sha(p) for p in [*a.params, a.start_crosstalk]},
           "rounds": rounds, "converged": converged,
           "final": {"saturation": rounds[-1]["saturation"], "crosstalk": rounds[-1]["crosstalk"]},
           "l2_flags": ["--fit-max-adc", "3000", "--fit-quiet-mev", str(QUIET_MEV)]}
    (a.out / "chain.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
