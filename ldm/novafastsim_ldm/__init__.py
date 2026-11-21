"""Thin LDM orchestration over BdNMC / bdnmc-nova / optional ldmgen.

Does not reimplement the detector simulation. Generators stay external.
"""

from . import chain

__all__ = ["chain"]
__version__ = "0.0.4"
