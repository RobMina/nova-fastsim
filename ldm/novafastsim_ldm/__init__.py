"""Thin LDM orchestration over BdNMC / bdnmc-nova / optional ldmgen.

Runs the stock detector simulation unchanged (the validation reference for novafastsim_detsim).
Generators stay external.
"""

from . import chain

__all__ = ["chain"]
__version__ = "0.0.4"
