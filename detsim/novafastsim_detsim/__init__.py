"""On-device, differentiable NOvA Near Detector simulation in JAX.

Particles -> energy-deposit segments (mean track and shower models) -> scintillator path lengths per cell ->
Birks quenching, looped-fibre attenuation, APD and front-end shaping -> per-cell expected response, plus a sampled
readout drawn around it. See docs/detsim.md.
"""

from .config import Detector, Geometry, Settings, default_params, from_dict, load, to_device
from .particles import Particles, from_bdnmc, from_hepevt, make
from .sim import Hits, Response, expected, expected_batch, sampled, sampled_batch

__all__ = [
    "Detector", "Geometry", "Settings", "default_params", "from_dict", "load", "to_device",
    "Particles", "from_bdnmc", "from_hepevt", "make",
    "Hits", "Response", "expected", "expected_batch", "sampled", "sampled_batch",
]
