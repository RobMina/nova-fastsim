"""Thin LDM orchestration over BdNMC / bdnmc-nova / optional ldmgen.

Does not reimplement TextFileGen. Generators stay external.
"""

from . import chain

__all__ = ["chain"]
__version__ = "0.0.4"
