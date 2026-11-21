"""Describe and drive the BdNMC JAX → HEPEVT → stock detector-simulation path."""

from __future__ import annotations

from dataclasses import dataclass

# Stages this package orchestrates. The detector simulation itself is stock collaboration
# software running in its own image.
STAGES = (
    "generate",   # BdNMC JAX via bdnmc_nova.generate → events.txt (+ optional HDF5)
    "ldmgen",     # optional: ldmgen convert/export when the package is installed
    "simulate",   # bdnmc_nova.jobspec → stock detector simulation (sim image)
)


@dataclass(frozen=True)
class PathInfo:
    """One documented hop in the LDM example chain."""

    stage: str
    owner: str
    notes: str


PATH = (
    PathInfo("generate", "bdnmc-nova + bdnmc-jax",
             "BdNMC JAX → HEPEVT text file (bdnmc_nova.generate)"),
    PathInfo("ldmgen", "ldmgen (optional; RobMina/ldmgen)",
             "ldmgen/1 HDF5 analysis record; export nova still uses the bdnmc-nova HEPEVT writer"),
    PathInfo("simulate", "stock detector simulation (collaboration software)",
             "HEPEVT → detector simulation → reco in the sim image; two-image with JAX host"),
)


def describe() -> str:
    """Human-readable chain summary for CLI and docs."""
    lines = ["nova-fastsim LDM example chain (orchestrate only):", ""]
    for i, p in enumerate(PATH, 1):
        lines.append(f"{i}. [{p.stage}] {p.owner}")
        lines.append(f"   {p.notes}")
    lines += ["", "The detector simulation is never reimplemented here."]
    return "\n".join(lines)
