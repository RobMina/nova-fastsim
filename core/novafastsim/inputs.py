"""Validate and check a nova-fastsim-inputs manifest (schema v1). No private files here."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

BLOCK = 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(s in ("", ".", "..") for s in value.split("/")):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def under(root: Path, name: str) -> Path:
    root = Path(root).resolve()
    dest = (root / relative(name)).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f"path escapes root: {name}")
    return dest


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def validate(obj: dict) -> dict:
    """Fail closed on a bad schema-v1 manifest object."""
    if not isinstance(obj, dict):
        raise ValueError("manifest must be a JSON object")
    if obj.get("schema_version") != 1:
        raise ValueError("expected schema_version=1")
    if not isinstance(obj.get("release"), str) or not obj["release"].strip():
        raise ValueError("release name required")
    files = obj.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("expected nonempty files list")
    seen = set()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("each file must be an object")
        name = relative(row.get("path", ""))
        if name in seen:
            raise ValueError(f"duplicate manifest path: {name}")
        seen.add(name)
        if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
            raise ValueError(f"invalid size: {name}")
        if not SHA256.fullmatch(row.get("sha256", "")):
            raise ValueError(f"invalid sha256: {name}")
        for key in ("provenance", "redistribution"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing {key}: {name}")
        if row["redistribution"] not in ("nova-internal", "approved"):
            raise ValueError(f"redistribution must be nova-internal or approved: {name}")
    return obj


def load_manifest(path: Path | str) -> dict:
    return validate(json.loads(Path(path).read_text()))


def file_status(root: Path, row: dict) -> str:
    path = under(root, row["path"])
    try:
        if not path.is_file():
            return "missing"
        if path.stat().st_size != row["size_bytes"]:
            return "size-mismatch"
        return "ok" if digest(path) == row["sha256"] else "hash-mismatch"
    except OSError as exc:
        return f"unreadable: {exc}"


def check(root: Path | str, manifest_path: Path | str) -> list[dict]:
    """Return per-file status rows against ``root`` (placeholders in RELEASE.example fail as missing)."""
    man = load_manifest(manifest_path)
    root_p = Path(root)
    return [{"path": row["path"], "status": file_status(root_p, row)} for row in man["files"]]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m novafastsim.inputs", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="validate a manifest JSON (no files on disk)")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("check", help="check files under a release root against a manifest")
    p.add_argument("manifest", type=Path)
    p.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "validate":
        man = load_manifest(args.manifest)
        print(json.dumps({"ok": True, "release": man["release"], "n_files": len(man["files"])}, indent=1))
        return 0
    rows = check(args.root, args.manifest)
    print(json.dumps(rows, indent=1))
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
