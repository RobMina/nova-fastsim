"""CLI: python -m novafastsim_ldm {describe,generate,submit,ldmgen-*,...}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novafastsim import dtype, site

from . import chain, generate, hepevt, ldmgen_bridge, submit


def _load_site(args):
    if args.site_env:
        site.load_dotenv(args.site_env)
    elif Path("site.env").is_file():
        site.load_dotenv("site.env")
    # Validate dtype after site.env so a bad value fails here, not at import.
    dtype.resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m novafastsim_ldm",
        description="Thin LDM orchestration for nova-fastsim (BdNMC → HEPEVT → stock detector simulation).",
    )
    ap.add_argument("--site-env", type=Path, help="KEY=value site file (default: ./site.env if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("describe", help="print the orchestration path")

    p = sub.add_parser("generate", help="BdNMC JAX → HEPEVT text file via bdnmc_nova.generate")
    p.add_argument("--card", type=Path)
    p.add_argument("--out", type=Path, help="events.txt (default NOVAFS_EVENTS_OUT or events.txt)")
    p.add_argument("--config", type=Path)
    p.add_argument("--hdf5", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--trials", type=int)
    p.add_argument("--channel", default="")
    p.add_argument("--dry-run", action="store_true", help="print resolved argv; do not run BdNMC")

    p = sub.add_parser("submit", help="submit two-stage job via bdnmc_nova.jobspec")
    p.add_argument("--card", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--handle")
    p.add_argument("--endpoint", help="hependpoints name (default NOVAFS_ENDPOINT)")
    p.add_argument("--stages", default="gen,sim", choices=("gen,sim", "gen", "sim"))
    p.add_argument("--gpus", type=int)
    p.add_argument("--precision", choices=("float32", "float64"))
    p.add_argument("--trials", type=int)
    p.add_argument("--events", type=Path, help="sim-only: local HEPEVT file to embed")
    p.add_argument("--events-at", default="", help="sim-only: HEPEVT path as seen on the site")
    p.add_argument("--max-events", type=int, help="sim-only: embed only the first N events of --events")
    p.add_argument("--script-out", type=Path,
                   help="write the job script only (dry-run; embeds private frame — mode 0700)")
    p.add_argument("--dry-run", action="store_true", help="validate site and print argv; do not submit")

    sub.add_parser("ldmgen-status", help="whether optional ldmgen / bdnmc-nova are importable")

    p = sub.add_parser("ldmgen-info", help="summarise an ldmgen/1 HDF5 record")
    p.add_argument("record", type=Path)

    p = sub.add_parser("ldmgen-export", help="ldmgen.export.to_nova (needs ldmgen + bdnmc-nova)")
    p.add_argument("record", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--max-events", type=int, default=-1)
    p.add_argument("--summary", type=Path)

    p = sub.add_parser("ldmgen-convert-jax", help="BdNMC JAX HDF5 → ldmgen/1 (needs ldmgen)")
    p.add_argument("hdf5", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--detector", default="nova-nd")

    p = sub.add_parser(
        "events-info",
        help="validate / summarise a HEPEVT input file (does not run the detector simulation)",
    )
    p.add_argument("events", type=Path, help="path to events.txt (or equivalent)")
    p.add_argument(
        "--max-list",
        type=int,
        default=20,
        help="list at most this many event summaries in JSON (default 20; 0 = counts only)",
    )

    p = sub.add_parser(
        "events-head",
        help="write the first N events of a HEPEVT file (orchestration helper)",
    )
    p.add_argument("events", type=Path)
    p.add_argument("--n", type=int, required=True, help="number of events to keep (<0 = all)")
    p.add_argument("--out", type=Path, required=True)

    args = ap.parse_args(argv)
    if args.cmd != "describe":
        _load_site(args)

    if args.cmd == "describe":
        print(chain.describe())
        return 0
    if args.cmd == "generate":
        result = generate.run(
            card=args.card, out=args.out, config=args.config, hdf5=args.hdf5,
            summary=args.summary, trials=args.trials, channel=args.channel or None,
            dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result, indent=1, default=str))
        return 0
    if args.cmd == "submit":
        result = submit.run(
            card=args.card, config=args.config, handle=args.handle, endpoint=args.endpoint,
            stages=args.stages, gpus=args.gpus, precision=args.precision, trials=args.trials,
            events=args.events, events_at=args.events_at or None, max_events=args.max_events,
            dry_run_script=args.script_out, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result, indent=1, default=str))
        return 0
    if args.cmd == "ldmgen-status":
        print(json.dumps(ldmgen_bridge.status(), indent=1))
        return 0
    if args.cmd == "ldmgen-info":
        print(json.dumps(ldmgen_bridge.info(args.record), indent=1, default=str))
        return 0
    if args.cmd == "ldmgen-export":
        info = ldmgen_bridge.export_nova(
            args.record, args.out, config=args.config, max_events=args.max_events, summary=args.summary)
        print(json.dumps(info, indent=1, default=str))
        return 0
    if args.cmd == "ldmgen-convert-jax":
        print(json.dumps(
            ldmgen_bridge.convert_bdnmc_jax(args.hdf5, args.out, detector=args.detector),
            indent=1, default=str))
        return 0
    if args.cmd == "events-info":
        info = hepevt.inspect(args.events, max_events=None)
        payload = info.to_dict()
        if args.max_list <= 0:
            payload["events"] = []
        else:
            payload["events"] = payload["events"][: args.max_list]
        print(json.dumps(payload, indent=1))
        return 0
    if args.cmd == "events-head":
        text = Path(args.events).read_text(encoding="utf-8")
        out = hepevt.head_text(text, args.n)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out, encoding="utf-8")
        print(json.dumps({"out": str(args.out), "bytes": len(out.encode("utf-8"))}, indent=1))
        return 0
    ap.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
