"""ND calibration libraries (scaffolding).

hep-simlib recorder → banks under the ``nova`` profile → ``hepsimlib.fit`` /
expected. Optional fibre / WLS via ``scint-fibre-fastsim`` (see ``fibre.status``).
"""

from . import banks, fibre

__version__ = "0.0.1"

__all__ = ["__version__", "banks", "fibre"]
