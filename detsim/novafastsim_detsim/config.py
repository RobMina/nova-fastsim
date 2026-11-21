"""Detector description: static geometry and settings (jit static arguments) and differentiable parameters.

The cell, its optics and the photodetector come from scint-fibre-fastsim: the cell cross-section and fibre default
to ``scintfibrefastsim.geometry.nova_cell_tdr()``, and the optical and APD parameters are that package's
``CellOptics`` and ``APDParams`` fields, keyed ``optics.<field>`` and ``apd.<field>`` in their units (mm, ns, MeV).
Everything else (planes, deposits) is in cm. The defaults are a public caricature of the Near Detector, not the
collaboration's geometry, materials or calibration; a private inputs bundle overrides them through :func:`load`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace

from scintfibrefastsim.backends.nova_apd import NOVA_APD, APDParams, Readout
from scintfibrefastsim.geometry import Cell, nova_cell_tdr
from scintfibrefastsim.response import NOVA_CELL_OPTICS, CellOptics

C_LIGHT = 29.9792458  # cm / ns
_CELL = nova_cell_tdr()
OPTICS_KEYS = tuple(f"optics.{k}" for k in CellOptics._fields)
APD_KEYS = tuple(f"apd.{k}" for k in APDParams._fields)


@dataclass(frozen=True)
class Geometry:
    """Planes of extruded cells, views alternating; lengths in cm, detector frame with z along the beam."""

    n_planes: int = 192
    n_cells: int = 96
    cell_width: float = _CELL.width / 10.0   # scintillator interior across the cell (u)
    cell_depth: float = _CELL.depth / 10.0   # scintillator interior along z
    web: float = _CELL.wall / 10.0           # inert wall between neighbouring cells
    face: float = 0.48        # inert front and back wall of a plane
    plane_gap: float = 0.0
    z_front: float = 0.0
    first_view: int = 0       # view of plane 0: 0 = cells along y measuring x, 1 = cells along x measuring y
    readout_end: tuple = (1, 1)  # per view: fibre ends at +w (1) or -w (-1)
    fibre_diameter: float = _CELL.fibre_diameter / 10.0
    looped: bool = _CELL.looped
    loop_extra: float = _CELL.loop_extra / 10.0

    @property
    def cell_pitch(self) -> float:
        return self.cell_width + self.web

    @property
    def plane_pitch(self) -> float:
        return self.cell_depth + 2.0 * self.face + self.plane_gap

    @property
    def cell_length(self) -> float:
        return self.n_cells * self.cell_pitch

    @property
    def half_width(self) -> float:
        return 0.5 * self.cell_length

    @property
    def z_back(self) -> float:
        return self.z_front + self.n_planes * self.plane_pitch

    @property
    def scint_fraction(self) -> float:
        """Scintillator volume fraction of the active region."""
        return self.cell_width * self.cell_depth / (self.cell_pitch * self.plane_pitch)

    def view(self, plane: int) -> int:
        return (plane + self.first_view) % 2

    def cell(self) -> Cell:
        """The scint-fibre-fastsim cell of this geometry (mm)."""
        return Cell(width=self.cell_width * 10.0, depth=self.cell_depth * 10.0, length=self.cell_length * 10.0,
                    wall=self.web * 10.0, fibre_diameter=self.fibre_diameter * 10.0, looped=self.looped,
                    loop_extra=self.loop_extra * 10.0)


@dataclass(frozen=True)
class Settings:
    """Discretisation and readout settings (static: they fix array shapes)."""

    n_long: int = 64          # longitudinal shower bins
    n_rad: int = 3            # radial quadrature nodes per lateral component (core, tail)
    n_phi: int = 6            # azimuthal nodes
    n_track: int = 256        # track segments per charged track
    track_step: float = 5.0   # cm per track segment
    max_segment: float = 15.0  # cm; sets the plane and cell windows of the geometry
    n_samples: int = 8        # readout samples after the hit
    tick: float = 125.0       # ns per readout sample
    dcs_lag: int = 3          # dual-correlated-sampling lag in samples

    def readout(self) -> Readout:
        return Readout(n_samples=self.n_samples, tick=self.tick, dcs_lag=self.dcs_lag)


@dataclass(frozen=True)
class Detector:
    """Geometry plus settings: the static half of a simulation call."""

    geometry: Geometry = field(default_factory=Geometry)
    settings: Settings = field(default_factory=Settings)

    @property
    def plane_window(self) -> int:
        return int(math.ceil(self.settings.max_segment / self.geometry.plane_pitch)) + 1

    @property
    def cell_window(self) -> int:
        return int(math.ceil(self.settings.max_segment / self.geometry.cell_pitch)) + 1

    @property
    def n_channels(self) -> int:
        return self.geometry.n_planes * self.geometry.n_cells


@dataclass(frozen=True)
class Material:
    """Bulk constants used to derive default shower and range parameters (PDG-style values)."""

    name: str
    density: float      # g / cm^3
    x0: float           # radiation length, g / cm^2
    ec: float           # critical energy for electrons, MeV
    stopping: float     # minimum-ionisation stopping power, MeV cm^2 / g


SCINTILLATOR = Material("liquid scintillator (CH2 caricature)", 0.859, 44.77, 93.1, 2.08)
INERT = Material("PVC", 1.30, 19.63, 50.0, 1.78)


def derived_params(geometry: Geometry = Geometry(), scint: Material = SCINTILLATOR,
                   inert: Material = INERT) -> dict:
    """Shower and range constants of the active-region mixture."""
    fs = geometry.scint_fraction
    ms, mi = fs * scint.density, (1.0 - fs) * inert.density
    ws, wi = ms / (ms + mi), mi / (ms + mi)
    x0_g = 1.0 / (ws / scint.x0 + wi / inert.x0)
    x0 = x0_g / (ms + mi)
    ec = ws * scint.ec + wi * inert.ec
    stop_inert = inert.stopping / scint.stopping
    return dict(
        rho_scint=scint.density,
        rho_inert=inert.density,
        stop_inert=stop_inert,
        rho_stop=ms + mi * stop_inert,
        x0=x0,
        ec=ec,
        rm=21.2052 * x0 / ec,
    )


def default_params(geometry: Geometry = Geometry()) -> dict:
    """Differentiable parameters (host floats); see docs/detsim.md for meanings and units."""
    p = derived_params(geometry)
    p.update(
        z_eff=7.0,            # effective Z of the lateral shower parametrisation
        shower_b=0.5,         # longitudinal gamma-profile slope b
        shower_dedx=2.0,      # MeV/cm, mean dE/dx of shower particles for Birks
        reco_scale=1.0,       # visible-to-true energy scale of reco.energy
    )
    p.update({f"optics.{k}": float(v) for k, v in NOVA_CELL_OPTICS._asdict().items()})
    p.update({f"apd.{k}": float(v) for k, v in NOVA_APD._asdict().items()})
    return p


def optics(params) -> CellOptics:
    """The ``CellOptics`` pytree of a parameter dict."""
    return CellOptics(*(params[k] for k in OPTICS_KEYS))


def apd(params) -> APDParams:
    """The ``APDParams`` pytree of a parameter dict."""
    return APDParams(*(params[k] for k in APD_KEYS))


def to_device(params: dict, dtype=None) -> dict:
    """Parameters as JAX arrays of the device float (``NOVA_DEVICE_FLOAT`` when ``dtype`` is None)."""
    import jax.numpy as jnp
    from novafastsim import dtype as nd

    import jax

    dt = nd.compute_dtype() if dtype is None else nd.compute_dtype(str(jnp.dtype(dtype)))
    if dt == jnp.float64 and not jax.config.jax_enable_x64:
        raise RuntimeError("float64 parameters need jax_enable_x64: call novafastsim.dtype.apply_jax_x64() first")
    def conv(v):
        if hasattr(v, "_fields"):                       # a NamedTuple table such as CellTable
            return type(v)(*(jnp.asarray(a, dt) for a in v))
        return jnp.asarray(v, dt)

    return {k: conv(v) for k, v in params.items()}


def _merge(cls, base, overrides: dict, where: str):
    names = {f.name for f in fields(cls)}
    bad = sorted(set(overrides) - names)
    if bad:
        raise KeyError(f"unknown {where} keys: {bad}")
    over = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    return replace(base, **over)


def from_dict(cfg: dict) -> tuple[Detector, dict]:
    """``{"geometry": {...}, "settings": {...}, "params": {...}}`` over the defaults; unknown keys raise."""
    bad = sorted(set(cfg) - {"geometry", "settings", "params", "comment"})
    if bad:
        raise KeyError(f"unknown detector-config sections: {bad}")
    geom = _merge(Geometry, Geometry(), cfg.get("geometry", {}), "geometry")
    sett = _merge(Settings, Settings(), cfg.get("settings", {}), "settings")
    params = default_params(geom)
    over = cfg.get("params", {})
    bad = sorted(set(over) - set(params) - {"channel_eff", "optics_table"})
    if bad:
        raise KeyError(f"unknown params keys: {bad}")
    params.update(over)
    return Detector(geom, sett), params


def load(path) -> tuple[Detector, dict]:
    """Read a detector-config JSON (e.g. from the private inputs bundle)."""
    with open(path) as f:
        return from_dict(json.load(f))


def to_dict(detector: Detector, params: dict) -> dict:
    """Inverse of :func:`from_dict` (scalar parameters only)."""
    return dict(geometry=asdict(detector.geometry), settings=asdict(detector.settings),
                params={k: float(v) for k, v in params.items() if k not in ("channel_eff", "optics_table")})
