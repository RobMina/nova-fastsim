"""Differentiable event-level reconstruction from per-cell charges: calorimetric energy and shower direction.

Inputs are per-cell charges in photoelectrons and per-cell weights: expected mode uses ``pe * p_hit``, a sampled
readout uses ``hit`` (charges ``adc / adc_per_pe``). Everything is a smooth function of the charges, so gradients
reach the simulation parameters and the particle kinematics through the expected response.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from scintfibrefastsim.response import fibre_transport

from . import config
from .config import Detector
from .geometry import cell_centres


class ViewFit(NamedTuple):
    centre: object   # weighted mean transverse coordinate, cm
    z: object        # weighted mean z, cm
    slope: object    # du/dz from the weighted regression
    weight: object   # total weight


def view_fits(det: Detector, w):
    """Weighted linear regression of u on z in each view: ``(x_view, y_view)``."""
    u, z, view = cell_centres(det)
    out = []
    for v in (0, 1):
        wv = w * (view == v)[:, None]
        W = jnp.sum(wv)
        Ws = jnp.where(W > 0, W, 1.0)
        zm = jnp.sum(wv * z[:, None]) / Ws
        um = jnp.sum(wv * u[None, :]) / Ws
        dz = z[:, None] - zm
        szz = jnp.sum(wv * dz * dz)
        szu = jnp.sum(wv * dz * (u[None, :] - um))
        slope = jnp.where(szz > 0, szu / jnp.where(szz > 0, szz, 1.0), 0.0)
        out.append(ViewFit(um, zm, slope, W))
    return tuple(out)


def direction(det: Detector, w):
    """Unit direction from the two view slopes."""
    fx, fy = view_fits(det, w)
    d = jnp.stack([fx.slope, fy.slope, jnp.ones_like(fx.slope)])
    return d / jnp.linalg.norm(d)


def theta2(det: Detector, w, beam=(0.0, 0.0, 1.0)):
    """Squared angle (rad^2) between the reconstructed direction and ``beam``."""
    b = jnp.asarray(beam, w.dtype)
    c = jnp.clip(jnp.dot(direction(det, w), b / jnp.linalg.norm(b)), -1.0, 1.0)
    return jnp.arccos(c) ** 2


def visible_energy(det: Detector, params, q):
    """Attenuation-corrected visible energy (MeV) of read-out charges ``q`` (PE): each view's cells are corrected
    at the other view's charge-weighted centroid."""
    g = det.geometry
    _, _, view = cell_centres(det)
    fx, fy = view_fits(det, q)
    w_along = jnp.where(view == 0, fy.centre, fx.centre)                        # [P]
    sign = jnp.where(view == 0, g.readout_end[0], g.readout_end[1])
    dist = jnp.clip(g.half_width - sign * w_along, 0.0, 2.0 * g.half_width)
    opt = config.optics(params)
    collect, _ = fibre_transport(dist * 10.0, g.cell(), opt)
    pe_per_mev = opt.yield_per_mev * params["apd.qe"]
    return jnp.sum(q / (pe_per_mev * collect[:, None]))


def energy(det: Detector, params, q):
    """Calorimetric energy (MeV): visible energy over ``reco_scale``."""
    return visible_energy(det, params, q) / params["reco_scale"]


def readout_charge(response=None, hits=None, params=None):
    """Read-out charge per cell (PE): ``pe * p_hit`` of an expected response, or ``adc / adc_per_pe`` of hits."""
    if hits is not None:
        return jnp.where(hits.hit, hits.adc / params["apd.adc_per_pe"], 0.0)
    return response.pe * response.p_hit
