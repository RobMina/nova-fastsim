"""hep-simbench stub for NOvA ND workflows.

When hep-simbench is installed, runs the synthetic dry-run layout so the ND track
can pin scoring / bench JSON shapes. Real ND hist converters come later (phase 3+).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def status() -> dict:
    try:
        import hepsimbench
        return {"hepsimbench": getattr(hepsimbench, "__version__", "unknown"), "available": True}
    except ImportError:
        return {
            "available": False,
            "note": "pip install 'hep-simbench @ git+https://github.com/RobMina/hep-simbench.git'",
        }


def synthetic(out: Path) -> dict:
    """Write hep-simbench's synthetic reference + submissions under ``out``."""
    from hepsimbench.synthetic import write

    out = Path(out)
    write(out)
    return {"out": str(out), "layout": "hepsimbench synthetic (good/shifted/slow)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("synthetic", help="write hep-simbench synthetic example (needs hep-simbench)")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    print(json.dumps(synthetic(args.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
