"""Load site.env: every path/host comes from the environment; missing names fail closed."""

from __future__ import annotations

import os
from pathlib import Path


def require(name: str) -> str:
    """Return ``os.environ[name]`` or raise with the variable name."""
    value = os.environ.get(name)
    if value is None or value == "":
        raise KeyError(f"missing required site variable {name}")
    return value


def load_dotenv(path: str | os.PathLike | None = None) -> Path:
    """Source a ``KEY=value`` file into ``os.environ`` (no shell). Does not override set keys."""
    env_path = Path(path) if path else Path("site.env")
    if not env_path.is_file():
        raise FileNotFoundError(f"site env not found: {env_path}")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val
    return env_path
