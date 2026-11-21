"""Primary particles to energy-deposit segments: expected (mean) models, differentiable in the kinematics.

- Charged tracks (mu, pi, K, p): straight line, mean continuous loss from the range table, column density
  ``rho_stop`` inside the detector envelope and none outside. No multiple scattering or hadronic interactions.
- Electromagnetic showers (e+-, photon): gamma-distribution longitudinal profile (Longo-Sestili form, t_max =
  ln(E/E_c) + C with C = -0.5 for electrons and +0.5 for photons) and a two-component radial profile of the
  Grindhammer-Peters form, both in units of the mixture's X0 and Moliere radius; integrated by a fixed quadrature
  whose node positions and weights move smoothly with energy, vertex and direction.

Every particle yields ``n_track`` track segments and ``n_long * 2 n_rad * n_phi`` shower segments; the ones its
model does not use carry zero energy, so shapes are fixed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc

from .config import C_LIGHT, Detector
from .geometry import box_interval
from .particles import Particles, mass_of, model_of, species_index

TRACK, SHOWER = 0, 1


class Segments(NamedTuple):
    p0: object     # [S, 3] cm
    p1: object     # [S, 3] cm
    edep: object   # [S] MeV deposited along the segment (inside the envelope)
    t0: object     # [S] ns
    t1: object     # [S] ns
    kind: object   # [S] TRACK or SHOWER
    mask: object   # [S] bool


def _unit(v):
    sq = jnp.sum(v * v, axis=-1, keepdims=True)
    pos = sq > 0
    n = jnp.sqrt(jnp.where(pos, sq, 1.0))
    return jnp.where(pos, v / n, jnp.array([0.0, 0.0, 1.0], v.dtype)), jnp.where(pos, n, 0.0)[..., 0]


def track_segments(det: Detector, params, tables, p: Particles) -> Segments:
    """``[N, n_track]`` segments of mean continuous loss for every particle (zero energy for non-tracks)."""
    s_ = det.settings
    dt = p.vertex.dtype
    T_tab, R_tab = (jnp.asarray(t, dt) for t in tables)
    sp = species_index(p.pdg)
    m = mass_of(p.pdg, dt)
    d, pmag = _unit(p.momentum)
    E = jnp.sqrt(pmag * pmag + m * m)
    T0 = E - m
    R0 = jax.vmap(jnp.interp)(T0, T_tab[sp], R_tab[sp])
    s = jnp.arange(s_.n_track + 1, dtype=dt) * s_.track_step                  # [n+1]
    b0, b1 = box_interval(det, p.vertex, d)                                    # along s (d is unit)
    inbox = jnp.maximum(jnp.minimum(s[None, :], b1[:, None]) - jnp.maximum(b0, 0.0)[:, None], 0.0)
    X = params["rho_stop"] * inbox
    R = jnp.maximum(R0[:, None] - X, 0.0)
    T = jax.vmap(jnp.interp)(R, R_tab[sp], T_tab[sp])
    edep = T[:, :-1] - T[:, 1:]
    beta = jnp.where(E > 0, pmag / jnp.where(E > 0, E, 1.0), 1.0)
    tt = p.time[:, None] + s[None, :] / (jnp.maximum(beta, 1e-3)[:, None] * C_LIGHT)
    pts = p.vertex[:, None, :] + s[None, :, None] * d[:, None, :]
    use = (model_of(p.pdg) == 2) & p.mask
    n = s_.n_track
    return Segments(pts[:, :-1], pts[:, 1:], jnp.where(use[:, None], edep, 0.0), tt[:, :-1], tt[:, 1:],
                    jnp.full((p.pdg.shape[0], n), TRACK, jnp.int32), jnp.broadcast_to(use[:, None], edep.shape))


def longitudinal(E, ec, b, photon):
    """``(a, t_max)`` of the gamma profile dE/dt = E b (bt)^(a-1) e^(-bt) / Gamma(a), t in X0."""
    y = jnp.maximum(E / ec, 1.0001)
    tmax = jnp.log(y) + jnp.where(photon, 0.5, -0.5)
    a = jnp.maximum(b * tmax + 1.0, 1.2)
    return a, tmax


def lateral(t, tmax, E_gev, z):
    """Grindhammer-Peters-form radial profile at depth ``t``: ``(R_core, R_tail, p_core)`` in Moliere radii."""
    lnE = jnp.log(jnp.maximum(E_gev, 1e-3))
    tau = t / jnp.maximum(tmax, 0.5)
    z1 = 0.0251 + 0.00319 * lnE
    z2 = 0.1162 - 0.000381 * z
    k1 = 0.659 - 0.00309 * z
    k2, k3 = 0.645, -2.59
    k4 = 0.3585 + 0.0421 * lnE
    p1 = 2.632 - 0.00094 * z
    p2 = 0.401 + 0.00187 * z
    p3 = 1.313 - 0.0686 * lnE
    r_c = z1 + z2 * tau
    r_t = k1 * (jnp.exp(k3 * (tau - k2)) + jnp.exp(k4 * (tau - k2)))
    q = (p2 - tau) / p3
    p = jnp.clip(p1 * jnp.exp(q - jnp.exp(q)), 0.0, 1.0)
    return r_c, r_t, p


def shower_segments(det: Detector, params, p: Particles) -> Segments:
    """``[N, n_long * 2 n_rad * n_phi]`` segments of the mean shower of every particle (zero for non-showers)."""
    s_ = det.settings
    dt = p.vertex.dtype
    nl, nr, nf = s_.n_long, s_.n_rad, s_.n_phi
    m = mass_of(p.pdg, dt)
    d, pmag = _unit(p.momentum)
    E = jnp.sqrt(pmag * pmag + m * m) - m                                       # [N] kinetic (photon: total)
    photon = p.pdg == 22
    x0, ec, rm, b = params["x0"], params["ec"], params["rm"], params["shower_b"]
    a, tmax = longitudinal(E, ec, b, photon)
    t_end = (a + 3.5 * jnp.sqrt(a) + 2.0) / b
    edges = t_end[:, None] * jnp.linspace(0.0, 1.0, nl + 1, dtype=dt)[None, :]   # [N, nl+1] in X0
    F = gammainc(a[:, None], jnp.maximum(b * edges, 1e-30))
    F = F.at[:, -1].set(1.0)
    dF = F[:, 1:] - F[:, :-1]                                                   # [N, nl]
    t_mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
    r_c, r_t, pc = lateral(t_mid, tmax[:, None], E[:, None] * 1e-3, params["z_eff"])
    q = (jnp.arange(nr, dtype=dt) + 0.5) / nr
    qf = jnp.sqrt(q / (1.0 - q))                                                # [nr]
    radius = jnp.concatenate([r_c[..., None] * qf, r_t[..., None] * qf], axis=-1) * rm   # [N, nl, 2nr] cm
    w_rad = jnp.concatenate([jnp.broadcast_to(pc[..., None], pc.shape + (nr,)),
                             jnp.broadcast_to((1.0 - pc)[..., None], pc.shape + (nr,))], axis=-1) / nr
    phi = 2.0 * jnp.pi * (jnp.arange(nf, dtype=dt) + 0.5) / nf
    helper = jnp.where(jnp.abs(d[:, 2:3]) < 0.9, jnp.array([0.0, 0.0, 1.0], dt), jnp.array([1.0, 0.0, 0.0], dt))
    e1, _ = _unit(jnp.cross(d, helper))
    e2 = jnp.cross(d, e1)
    off = (radius[..., None, None] * (jnp.cos(phi)[:, None] * e1[:, None, None, None, :]
                                       + jnp.sin(phi)[:, None] * e2[:, None, None, None, :]))  # [N, nl, 2nr, nf, 3]
    z0 = edges[:, :-1, None, None, None] * x0 * d[:, None, None, None, :]
    z1 = edges[:, 1:, None, None, None] * x0 * d[:, None, None, None, :]
    v = p.vertex[:, None, None, None, :]
    p0 = v + z0 + off
    p1 = v + z1 + off
    edep = E[:, None, None, None] * dF[:, :, None, None] * w_rad[..., None] / nf
    edep = jnp.broadcast_to(edep, p0.shape[:-1])
    tt0 = p.time[:, None] + edges[:, :-1] * x0 / C_LIGHT
    tt1 = p.time[:, None] + edges[:, 1:] * x0 / C_LIGHT
    use = (model_of(p.pdg) == 1) & p.mask
    N = p.pdg.shape[0]
    S = nl * 2 * nr * nf
    bc = lambda x: jnp.broadcast_to(x[:, :, None, None], (N, nl, 2 * nr, nf)).reshape(N, S)
    return Segments(p0.reshape(N, S, 3), p1.reshape(N, S, 3), jnp.where(use[:, None], edep.reshape(N, S), 0.0),
                    bc(tt0), bc(tt1), jnp.full((N, S), SHOWER, jnp.int32), jnp.broadcast_to(use[:, None], (N, S)))


def from_particles(det: Detector, params, tables, p: Particles) -> Segments:
    """All segments of one event, flattened to ``[S]``."""
    tr = track_segments(det, params, tables, p)
    sh = shower_segments(det, params, p)

    def cat(a, b):
        return jnp.concatenate([a.reshape((-1,) + a.shape[2:]), b.reshape((-1,) + b.shape[2:])], axis=0)

    return Segments(*(cat(a, b) for a, b in zip(tr, sh)))
