"""Describe and drive the BdNMC JAX → HEPEVT → stock TextFileGen path."""

from __future__ import annotations

from dataclasses import dataclass

# Stages this package orchestrates. TextFileGen itself is stock NOvASoft (sim image).
STAGES = (
    "generate",   # BdNMC JAX via bdnmc_nova.generate → events.txt (+ optional HDF5)
    "ldmgen",     # optional: ldmgen convert/export when the package is installed
    "simulate",   # bdnmc_nova.jobspec → NOvASoft TextFileGen in SL7
)


@dataclass(frozen=True)
class PathInfo:
    """One documented hop in the LDM example chain."""

    stage: str
    owner: str
    notes: str


PATH = (
    PathInfo("generate", "bdnmc-nova + bdnmc-jax",
             "BdNMC JAX → HEPEVT/TextFileGen file (bdnmc_nova.generate / textfilegen)"),
    PathInfo("ldmgen", "ldmgen (optional, not yet public)",
             "ldmgen/1 HDF5 analysis record; export nova still calls bdnmc_nova.textfilegen"),
    PathInfo("simulate", "NOvASoft TextFileGen (stock)",
             "SL7 art job: HEPEVT → MCTruth → G4 → reco; two-image with JAX host"),
)


def describe() -> str:
    """Human-readable chain summary for CLI and docs."""
    lines = ["nova-fastsim LDM example chain (orchestrate only):", ""]
    for i, p in enumerate(PATH, 1):
        lines.append(f"{i}. [{p.stage}] {p.owner}")
        lines.append(f"   {p.notes}")
    lines += ["", "TextFileGen is never reimplemented here."]
    return "\n".join(lines)
