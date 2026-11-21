"""CLI: python -m novafastsim_calib {describe,status,ensure-dirs}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novafastsim import dtype, site

from . import banks, fibre


def _load_site(args):
    if args.site_env:
        site.load_dotenv(args.site_env)
    elif Path("site.env").is_file():
        site.load_dotenv("site.env")
    dtype.resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m novafastsim_calib",
        description="NOvA ND calib scaffolding (hep-simlib banks under the nova profile).",
    )
    ap.add_argument("--site-env", type=Path, help="KEY=value site file (default: ./site.env if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("describe", help="print the calib bank plan")

    p = sub.add_parser("status", help="JSON: bank layout + optional scint-fibre probe")
    p.add_argument("--cache-dir", help="override NOVA_LIBRARY_CACHE for this invocation")
    p.add_argument("--precision", choices=("float32", "float64"))

    p = sub.add_parser("ensure-dirs", help="create empty subsystem dirs under the library cache")
    p.add_argument("--cache-dir", help="override NOVA_LIBRARY_CACHE for this invocation")
    p.add_argument("--precision", choices=("float32", "float64"))

    args = ap.parse_args(argv)
    if args.cmd != "describe":
        _load_site(args)

    if args.cmd == "describe":
        # describe does not require site.env; still show dtype default.
        print(banks.describe())
        return 0
    if args.cmd == "status":
        lay = banks.layout(cache_dir=args.cache_dir, precision=args.precision)
        print(json.dumps({"banks": lay.to_dict(), "fibre": fibre.status()}, indent=1))
        return 0
    if args.cmd == "ensure-dirs":
        lay = banks.layout(cache_dir=args.cache_dir, precision=args.precision)
        created = banks.ensure_dirs(lay)
        print(json.dumps({"layout": lay.to_dict(), "dirs": created}, indent=1))
        return 0
    ap.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
