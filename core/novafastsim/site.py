"""Load site.env: every path/host comes from the environment; missing names fail closed."""

from __future__ import annotations

import os
import re
from pathlib import Path

_EXPORT = re.compile(r"^export\s+", re.IGNORECASE)
_UNQUOTED_COMMENT = re.compile(r"\s+#.*$")


def require(name: str) -> str:
    """Return ``os.environ[name]`` or raise with the variable name."""
    value = os.environ.get(name)
    if value is None or value == "":
        raise KeyError(f"missing required site variable {name}")
    return value


def _parse_value(raw: str) -> str:
    """Parse a dotenv value: strip matching quotes; drop unquoted trailing ``#`` comments."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return _UNQUOTED_COMMENT.sub("", s).rstrip()


def load_dotenv(path: str | os.PathLike | None = None) -> Path:
    """Source a ``KEY=value`` file into ``os.environ`` (no shell).

    Rules (closest to common dotenv + shell expectations):

    - Real process environment wins over the file (never override a non-empty
      env value already present).
    - Within one file, the **last** occurrence of a key wins.
    - ``KEY=`` (empty) means unset: remove the key if we would have set it from
      the file, and do not insert ``\"\"``.
    - Leading ``export `` is accepted; unquoted trailing `` # comment`` is stripped.
    - Only matching quote pairs are stripped (no loose ``.strip(\"'\")``).
    """
    env_path = Path(path) if path else Path("site.env")
    if not env_path.is_file():
        raise FileNotFoundError(f"site env not found: {env_path}")

    parsed: dict[str, str | None] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = _EXPORT.sub("", line, count=1)
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or any(c.isspace() for c in key):
            continue
        value = _parse_value(val)
        parsed[key] = value if value != "" else None

    for key, value in parsed.items():
        existing = os.environ.get(key)
        if existing is not None and existing != "":
            continue  # real environment wins
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return env_path
