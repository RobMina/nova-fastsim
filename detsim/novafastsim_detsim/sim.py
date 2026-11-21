"""The on-device detector simulation: particles -> segments -> cell overlaps -> per-cell response.

Deposits and the plane/cell geometry are this package's; the cell response is scint-fibre-fastsim's: quenching and
looped-fibre transport (``scintfibrefastsim.response.light``, analytic or a ``CellTable`` passed as
``params["optics_table"]``) and the NOvA APD + front end (``scintfibrefastsim.backends.nova_apd``).

``expected`` is deterministic and differentiable in the parameters and the particle kinematics; ``sampled`` adds
the stochastic readout on top of it. Both are jitted with the ``Detector`` static and take one event; batch with
``jax.vmap(f, in_axes=(None, None, 0))`` (see :func:`expected_batch`).
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from scintfibrefastsim.backends import nova_apd
from scintfibrefastsim.response import light as cell_light

from . import config, deposits, geometry
from .config import Detector
from .particles import Particles
from .ranges import tables as _range_tables


class Response(NamedTuple):
    """Per-cell expected response ``[n_planes, n_cells]`` and event diagnostics."""

    edep: object    # MeV in scintillator
    evis: object    # MeV after quenching
    pe: object      # expected photoelectrons (photons at the pixel x APD quantum efficiency)
    time: object    # ns, light-weighted mean arrival
    adc: object     # expected DCS peak
    p_hit: object   # probability that the cell is read out
    diag: dict      # e_total, e_scint, e_inert, e_outside, overflow


class Hits(NamedTuple):
    """One sampled readout, ``[n_planes, n_cells]``."""

    pe: object
    adc: object
    time: object
    hit: object


@lru_cache(maxsize=None)
def _tables_np():
    return _range_tables()


def tables(dtype):
    T, R = _tables_np()
    return jnp.asarray(T, dtype), jnp.asarray(R, dtype)


def accumulate(det: Detector, params, seg: deposits.Segments) -> Response:
    """Segments to the expected per-cell response."""
    g = det.geometry
    P, C = g.n_planes, g.n_cells
    pc = geometry.pieces(det, seg.p0, seg.p1)
    l_scint = jnp.sum(pc.length, axis=-1)
    l_inert = jnp.maximum(pc.len_box - l_scint, 0.0)
    rs, ri = params["rho_scint"], params["rho_inert"] * params["stop_inert"]
    denom = rs * l_scint + ri * l_inert
    has = (denom > 0) & seg.mask
    e_unit = jnp.where(has, seg.edep / jnp.where(has, denom, 1.0), 0.0)        # MeV per scint-equivalent g/cm^2
    e_piece = e_unit[:, None] * rs * pc.length
    dedx = jnp.where(seg.kind == deposits.SHOWER, params["shower_dedx"], e_unit * rs)          # MeV / cm
    opt, ap = config.optics(params), config.apd(params)
    photons, delay = cell_light(e_piece, jnp.broadcast_to(dedx[:, None] / 10.0, e_piece.shape), pc.distance * 10.0,
                                g.cell(), opt, params.get("optics_table"))
    v_piece = e_piece / (1.0 + opt.birks_kb * dedx[:, None] / 10.0 + opt.birks_kc * (dedx[:, None] / 10.0) ** 2)
    light = nova_apd.photoelectrons(photons, ap)
    if "channel_eff" in params:
        light = light * jnp.reshape(params["channel_eff"], (-1,))[pc.index]
    t_piece = seg.t0[:, None] + (seg.t1 - seg.t0)[:, None] * pc.t_mid + delay

    idx = pc.index.reshape(-1)

    def scatter(v):
        return jnp.zeros(P * C, v.dtype).at[idx].add(v.reshape(-1)).reshape(P, C)

    edep, evis, pe, pet = scatter(e_piece), scatter(v_piece), scatter(light), scatter(light * t_piece)
    lit = pe > 0
    time = jnp.where(lit, pet / jnp.where(lit, pe, 1.0), 0.0)
    adc, p_hit = nova_apd.expected(pe, time, ap, det.settings.readout())
    m = seg.mask
    diag = dict(
        e_total=jnp.sum(jnp.where(m, seg.edep, 0.0)),
        e_scint=jnp.sum(edep),
        e_inert=jnp.sum(e_unit * ri * l_inert),
        e_outside=jnp.sum(jnp.where(m & ~has, seg.edep, 0.0)),
        overflow=jnp.sum(m & pc.overflow & (seg.edep > 0)),
    )
    return Response(edep, evis, pe, time, adc, p_hit, diag)


@partial(jax.jit, static_argnums=0)
def expected(det: Detector, params, particles: Particles) -> Response:
    """Expected response of one event (``particles`` ``[N]``)."""
    seg = deposits.from_particles(det, params, tables(particles.vertex.dtype), particles)
    return accumulate(det, params, seg)


@partial(jax.jit, static_argnums=0)
def sampled(det: Detector, params, particles: Particles, key) -> tuple[Response, Hits]:
    """Expected response and one sampled readout of one event."""
    r = expected(det, params, particles)
    return r, Hits(*nova_apd.sample(key, r.pe, r.time, config.apd(params), det.settings.readout()))


def expected_batch(det: Detector, params, particles: Particles) -> Response:
    """``expected`` over a leading event axis."""
    return jax.vmap(lambda p: expected(det, params, p))(particles)


def sampled_batch(det: Detector, params, particles: Particles, key) -> tuple[Response, Hits]:
    """``sampled`` over a leading event axis, one key per event."""
    keys = jax.random.split(key, particles.pdg.shape[0])
    return jax.vmap(lambda p, k: sampled(det, params, p, k))(particles, keys)


def to_numpy(x):
    return jax.tree_util.tree_map(np.asarray, x)
