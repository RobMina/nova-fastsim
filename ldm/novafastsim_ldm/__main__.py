"""CLI: python -m novafastsim_ldm {describe,generate,submit,ldmgen-status,ldmgen-export,...}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novafastsim import site

from . import chain, generate, ldmgen_bridge, submit


def _load_site(args):
    if args.site_env:
        site.load_dotenv(args.site_env)
    elif Path("site.env").is_file():
        site.load_dotenv("site.env")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m novafastsim_ldm",
        description="Thin LDM orchestration for nova-fastsim (BdNMC → HEPEVT → stock TextFileGen).",
    )
    ap.add_argument("--site-env", type=Path, help="KEY=value site file (default: ./site.env if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("describe", help="print the orchestration path")

    p = sub.add_parser("generate", help="BdNMC JAX → TextFileGen file via bdnmc_nova.generate")
    p.add_argument("--card", type=Path)
    p.add_argument("--out", type=Path, help="events.txt (default NOVAFS_EVENTS_OUT or events.txt)")
    p.add_argument("--config", type=Path)
    p.add_argument("--hdf5", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--trials", type=int)
    p.add_argument("--channel", default="")

    p = sub.add_parser("submit", help="submit two-stage job via bdnmc_nova.jobspec")
    p.add_argument("--card", type=Path)
    p.add_argument("--handle")
    p.add_argument("--endpoint", help="hependpoints name (default NOVAFS_ENDPOINT)")
    p.add_argument("--stages", default="gen,sim", choices=("gen,sim", "gen", "sim"))
    p.add_argument("--gpus", type=int)
    p.add_argument("--precision", choices=("float32", "float64"))
    p.add_argument("--trials", type=int)
    p.add_argument("--script-out", type=Path,
                   help="write the job script only (dry-run; embeds private frame — mode 0700)")

    sub.add_parser("ldmgen-status", help="whether optional ldmgen is importable")

    p = sub.add_parser("ldmgen-export", help="ldmgen export nova (requires ldmgen installed)")
    p.add_argument("record", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)

    p = sub.add_parser("ldmgen-convert-jax", help="ldmgen convert bdnmc-jax (requires ldmgen)")
    p.add_argument("hdf5", type=Path)
    p.add_argument("--out", type=Path, required=True)

    args = ap.parse_args(argv)
    if args.cmd != "describe":
        _load_site(args)

    if args.cmd == "describe":
        print(chain.describe())
        return 0
    if args.cmd == "generate":
        generate.run(card=args.card, out=args.out, config=args.config, hdf5=args.hdf5,
                     summary=args.summary, trials=args.trials, channel=args.channel or None)
        return 0
    if args.cmd == "submit":
        submit.run(card=args.card, handle=args.handle, endpoint=args.endpoint, stages=args.stages,
                   gpus=args.gpus, precision=args.precision, trials=args.trials,
                   dry_run_script=args.script_out)
        return 0
    if args.cmd == "ldmgen-status":
        print(json.dumps(ldmgen_bridge.status(), indent=1))
        return 0
    if args.cmd == "ldmgen-export":
        ldmgen_bridge.export_nova(args.record, args.out, config=args.config)
        return 0
    if args.cmd == "ldmgen-convert-jax":
        ldmgen_bridge.convert_bdnmc_jax(args.hdf5, args.out)
        return 0
    ap.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
