"""Inspect HEPEVT *input* text files for detector simulation.

The detector simulation that reads these files is stock collaboration software and is never
reimplemented here. These helpers only read the HEPEVT-style text that
the bdnmc-nova HEPEVT writer / ``ldmgen export nova`` write, so orchestration can
validate a file before ``submit --stages sim``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventSummary:
    """One HEPEVT event: header index and particle count."""

    index: int
    n_particles: int


@dataclass(frozen=True)
class FileInfo:
    """Summary of a HEPEVT input file."""

    path: str
    n_events: int
    n_particles: int
    events: tuple[EventSummary, ...]
    bytes: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


def iter_events(text: str):
    """Yield ``(event_index, particle_field_rows)``; skip blank and ``#`` lines.

    Raises ``ValueError`` on a malformed header or truncated particle block.
    """
    lines = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        head = lines[i]
        if len(head) != 2:
            raise ValueError(
                f"expected HEPEVT event header '<event> <particles>', got {' '.join(head)!r}"
            )
        try:
            index = int(head[0])
            n = int(head[1])
        except ValueError as e:
            raise ValueError(f"non-integer event header: {' '.join(head)!r}") from e
        if n < 0:
            raise ValueError(f"negative particle count in header: {' '.join(head)!r}")
        if i + 1 + n > len(lines):
            raise ValueError(
                f"event {index}: header claims {n} particles but only {len(lines) - i - 1} lines remain"
            )
        yield index, lines[i + 1:i + 1 + n]
        i += 1 + n


def count_events(text: str) -> int:
    """Number of events in HEPEVT text (validates structure)."""
    return sum(1 for _ in iter_events(text))


def validate(text: str) -> FileInfo:
    """Parse and validate HEPEVT text; return a summary (path left empty)."""
    events: list[EventSummary] = []
    n_particles = 0
    for index, parts in iter_events(text):
        events.append(EventSummary(index=index, n_particles=len(parts)))
        n_particles += len(parts)
    return FileInfo(
        path="",
        n_events=len(events),
        n_particles=n_particles,
        events=tuple(events),
        bytes=len(text.encode("utf-8")),
    )


def inspect(path: Path | str, *, max_events: int | None = None) -> FileInfo:
    """Read ``path`` and return a validated summary.

    When ``max_events`` is set, only the first N events are retained in
    ``FileInfo.events`` (full-file counts still reflect the whole file).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    info = validate(text)
    events = info.events
    if max_events is not None:
        if max_events < 0:
            raise ValueError("max_events must be >= 0")
        events = events[:max_events]
    return FileInfo(
        path=str(p),
        n_events=info.n_events,
        n_particles=info.n_particles,
        events=events,
        bytes=p.stat().st_size,
    )


def head_text(text: str, n: int) -> str:
    """Return the first ``n`` events of HEPEVT text (all of them for ``n < 0``).

    Prefers bdnmc-nova's own ``head`` helper when that package is installed so
    behaviour matches the writer; otherwise uses a local parser.
    """
    if n < 0:
        return text
    try:
        from bdnmc_nova import hepevt as bn_hepevt

        return bn_hepevt.head(text, n)
    except ImportError:
        pass
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    events = 0
    i = 0
    while i < len(lines) and events < n:
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            out.append(lines[i])
            i += 1
            continue
        fields = lines[i].split()
        if len(fields) != 2:
            raise ValueError(
                f"line {i + 1}: expected an event header '<event> <particles>', got {lines[i]!r}"
            )
        k = int(fields[1])
        out += lines[i:i + 1 + k]
        i += 1 + k
        events += 1
    return "".join(out)
