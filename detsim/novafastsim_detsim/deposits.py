"""Primary particles to energy-deposit segments: expected (mean) models, differentiable in the kinematics.

- Charged tracks (mu, pi, K, p): straight line, mean continuous loss from the range table, column density
  ``rho_stop`` inside the plane mixture (the geometry's region boxes), ``rho_abs * stop_abs`` inside absorber boxes
  and none elsewhere. No multiple scattering or hadronic interactions.
- Electromagnetic showers (e+-, photon): gamma-distribution longitudinal profile (Longo-Sestili form, t_max =
  ln(E/E_c) + C + ``shower_tmax_shift`` with C = -0.5 for electrons and +0.5 for photons) and a two-component radial
  profile of the Grindhammer-Peters form, both in units of the mixture's X0 and Moliere radius; integrated by a fixed
  quadrature whose node positions and weights move smoothly with energy, vertex and direction. An electron's own
  ionisation before the shower develops (the "head") is a uniform on-axis deposit of E_c per X0 over the first
  ``head_x0`` radiation lengths, taken from the shower's energy. ``lat_core`` and ``lat_tail`` scale the two radial
  widths and ``lat_pcore`` the core fraction; all four extras are neutral at their defaults (0, 0, 1, 1, 1).

Every particle yields ``n_track`` track segments and ``n_long * 2 n_rad * n_phi`` shower segments; the ones its
model does not use carry zero energy, so shapes are fixed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc

from .config import C_LIGHT, Detector
from .geometry import length_in
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
    g = det.geometry
    v = p.vertex[:, None, :]
    dd = d[:, None, :]
    in_mix = length_in(g.region_boxes(), v, dd, 0.0, s[None, :])                 # [N, n+1] cm (d is unit)
    in_abs = length_in(g.absorbers, v, dd, 0.0, s[None, :])
    X = params["rho_stop"] * in_mix + params["rho_abs"] * params["stop_abs"] * in_abs
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


def longitudinal(E, ec, b, photon, shift=0.0):
    """``(a, t_max)`` of the gamma profile dE/dt = E b (bt)^(a-1) e^(-bt) / Gamma(a), t in X0."""
    y = jnp.maximum(E / ec, 1.0001)
    tmax = jnp.log(y) + jnp.where(photon, 0.5, -0.5) + shift
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


class Profile(NamedTuple):
    """The mean shower of every particle: longitudinal bins and radial parameters ``[N, n_long]``, axis frame."""

    E: object        # [N] shower energy (kinetic; photon total), MeV
    d: object        # [N, 3] axis
    e1: object       # [N, 3] transverse unit vectors
    e2: object
    edges: object    # [N, nl + 1] depth bin edges, X0
    dF: object       # [N, nl] energy fraction per bin
    r_c: object      # [N, nl] core radius, Moliere radii
    r_t: object      # [N, nl] tail radius, Moliere radii
    pc: object       # [N, nl] core fraction
    use: object      # [N] showering particles


def shower_profile(det: Detector, params, p: Particles, tmax_noise=None) -> Profile:
    """The longitudinal and radial profile of every particle's mean shower. ``tmax_noise`` ``[N]`` (X0) shifts each
    shower's maximum (the stochastic tier's per-shower fluctuation)."""
    s_ = det.settings
    dt = p.vertex.dtype
    nl = s_.n_long
    m = mass_of(p.pdg, dt)
    d, pmag = _unit(p.momentum)
    E = jnp.sqrt(pmag * pmag + m * m) - m                                       # [N] kinetic (photon: total)
    photon = p.pdg == 22
    ec, b = params["ec"], params["shower_b"]
    shift = params["shower_tmax_shift"] + (0.0 if tmax_noise is None else tmax_noise)
    a, tmax = longitudinal(E, ec, b, photon, shift)
    th = jnp.maximum(params["head_x0"], 0.0)
    t_end = jnp.maximum((a + 3.5 * jnp.sqrt(a) + 2.0) / b, 1.5 * th)
    edges = t_end[:, None] * jnp.linspace(0.0, 1.0, nl + 1, dtype=dt)[None, :]   # [N, nl+1] in X0
    F = gammainc(a[:, None], jnp.maximum(b * edges, 1e-30))
    F = F.at[:, -1].set(1.0)
    dG = F[:, 1:] - F[:, :-1]                                                   # [N, nl] gamma profile
    # electron head: E_c per X0 on the axis over [0, head_x0], at most half the energy
    e_head = jnp.where(jnp.abs(p.pdg) == 11, jnp.minimum(ec * th, 0.5 * E), 0.0)
    fh = jnp.where(E > 0, e_head / jnp.where(E > 0, E, 1.0), 0.0)[:, None]
    th_s = jnp.maximum(th, 1e-6)
    dH = jnp.clip(jnp.minimum(edges[:, 1:], th_s) - edges[:, :-1], 0.0, None) / th_s
    dF = (1.0 - fh) * dG + fh * dH
    t_mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
    r_c, r_t, pc = lateral(t_mid, tmax[:, None], E[:, None] * 1e-3, params["z_eff"])
    r_c, r_t = r_c * params["lat_core"], r_t * params["lat_tail"]
    pc = jnp.clip(pc * params["lat_pcore"], 0.0, 1.0)
    pos = dF > 0
    pc = jnp.where(pos, ((1.0 - fh) * dG * pc + fh * dH) / jnp.where(pos, dF, 1.0), pc)   # the head is core
    helper = jnp.where(jnp.abs(d[:, 2:3]) < 0.9, jnp.array([0.0, 0.0, 1.0], dt), jnp.array([1.0, 0.0, 0.0], dt))
    e1, _ = _unit(jnp.cross(d, helper))
    e2 = jnp.cross(d, e1)
    use = (model_of(p.pdg) == 1) & p.mask
    return Profile(E, d, e1, e2, edges, dF, r_c, r_t, pc, use)


def shower_segments(det: Detector, params, p: Particles, prof: Profile | None = None, scale=None) -> Segments:
    """``[N, n_long * 2 n_rad * n_phi]`` segments of the mean shower of every particle (zero for non-showers);
    ``scale`` ``[N]`` multiplies each particle's energy."""
    s_ = det.settings
    dt = p.vertex.dtype
    nl, nr, nf = s_.n_long, s_.n_rad, s_.n_phi
    prof = shower_profile(det, params, p) if prof is None else prof
    x0, rm = params["x0"], params["rm"]
    E, d, e1, e2, edges, dF = prof.E, prof.d, prof.e1, prof.e2, prof.edges, prof.dF
    q = (jnp.arange(nr, dtype=dt) + 0.5) / nr
    qf = jnp.sqrt(q / (1.0 - q))                                                # [nr]
    radius = jnp.concatenate([prof.r_c[..., None] * qf, prof.r_t[..., None] * qf], axis=-1) * rm   # [N, nl, 2nr] cm
    pc = prof.pc
    w_rad = jnp.concatenate([jnp.broadcast_to(pc[..., None], pc.shape + (nr,)),
                             jnp.broadcast_to((1.0 - pc)[..., None], pc.shape + (nr,))], axis=-1) / nr
    phi = 2.0 * jnp.pi * (jnp.arange(nf, dtype=dt) + 0.5) / nf
    off = (radius[..., None, None] * (jnp.cos(phi)[:, None] * e1[:, None, None, None, :]
                                       + jnp.sin(phi)[:, None] * e2[:, None, None, None, :]))  # [N, nl, 2nr, nf, 3]
    z0 = edges[:, :-1, None, None, None] * x0 * d[:, None, None, None, :]
    z1 = edges[:, 1:, None, None, None] * x0 * d[:, None, None, None, :]
    v = p.vertex[:, None, None, None, :]
    p0 = v + z0 + off
    p1 = v + z1 + off
    Es = E if scale is None else E * scale
    edep = Es[:, None, None, None] * dF[:, :, None, None] * w_rad[..., None] / nf
    edep = jnp.broadcast_to(edep, p0.shape[:-1])
    tt0 = p.time[:, None] + edges[:, :-1] * x0 / C_LIGHT
    tt1 = p.time[:, None] + edges[:, 1:] * x0 / C_LIGHT
    use = prof.use
    N = p.pdg.shape[0]
    S = nl * 2 * nr * nf
    bc = lambda x: jnp.broadcast_to(x[:, :, None, None], (N, nl, 2 * nr, nf)).reshape(N, S)
    return Segments(p0.reshape(N, S, 3), p1.reshape(N, S, 3), jnp.where(use[:, None], edep.reshape(N, S), 0.0),
                    bc(tt0), bc(tt1), jnp.full((N, S), SHOWER, jnp.int32), jnp.broadcast_to(use[:, None], (N, S)))


Q_RAD_MAX = 0.96   # radial quantile cut of the sampled spots: their mean radius matches the n_rad = 3 quadrature


def stochastic_segments(det: Detector, params, p: Particles, prof: Profile, key) -> Segments:
    """``[N, n_stoch]`` straight charged-track pieces carrying the fraction ``stoch_frac`` of each shower's energy.

    The track-length picture of a low-Z shower: the energy is deposited by charged tracks crossing a few cells each,
    not spread thinly. Each piece has length ``stoch_len`` (cm, at most ``max_segment``) and carries ``stoch_dedx``
    MeV per cm (the count follows the energy; above ``n_stoch`` pieces their energy grows instead). Its centre is
    drawn from the mean shower's longitudinal and radial profile (inverse CDFs, so the draws are differentiable in the
    shower parameters at fixed random numbers) and its direction is tilted from the axis by a Rayleigh angle of scale
    ``stoch_theta`` (rad). Pieces are kind TRACK: quenching uses their own dE/dx and the minimum-ionisation inert
    stopping."""
    s_ = det.settings
    dt = p.vertex.dtype
    n = s_.n_stoch
    N = p.pdg.shape[0]
    x0, rm = params["x0"], params["rm"]
    ku = jax.random.split(key, 6)
    u = [jax.random.uniform(k, (N, n), dt, minval=1e-6, maxval=1.0 - 1e-6) for k in ku]
    cdf = jnp.concatenate([jnp.zeros((N, 1), dt), jnp.cumsum(prof.dF, axis=-1)], axis=-1)
    cdf = cdf / jnp.maximum(cdf[:, -1:], 1e-30)
    t = jax.vmap(jnp.interp)(u[0], cdf, prof.edges)                             # [N, n] depth, X0
    nl = prof.dF.shape[1]
    span = jnp.maximum(prof.edges[:, -1:], 1e-12)
    k = jnp.clip(jnp.floor(t / span * nl).astype(jnp.int32), 0, nl - 1)
    r_c = jnp.take_along_axis(prof.r_c, k, axis=-1)
    r_t = jnp.take_along_axis(prof.r_t, k, axis=-1)
    pc = jnp.take_along_axis(prof.pc, k, axis=-1)
    R = jnp.where(u[1] < pc, r_c, r_t) * rm
    q = u[2] * Q_RAD_MAX
    r = R * jnp.sqrt(q / (1.0 - q))
    phi = 2.0 * jnp.pi * u[3]
    e1, e2, d = prof.e1[:, None, :], prof.e2[:, None, :], prof.d[:, None, :]
    centre = (p.vertex[:, None, :] + (t * x0)[..., None] * d
              + r[..., None] * (jnp.cos(phi)[..., None] * e1 + jnp.sin(phi)[..., None] * e2))
    theta = params["stoch_theta"] * jnp.sqrt(-2.0 * jnp.log(u[4]))
    psi = 2.0 * jnp.pi * u[5]
    dirn = (jnp.cos(theta)[..., None] * d
            + jnp.sin(theta)[..., None] * (jnp.cos(psi)[..., None] * e1 + jnp.sin(psi)[..., None] * e2))
    ell = jnp.clip(params["stoch_len"], 0.1, 0.95 * s_.max_segment)
    p0 = centre - 0.5 * ell * dirn
    p1 = centre + 0.5 * ell * dirn
    e_s = prof.E * params["stoch_frac"]                                          # [N]
    n_f = jnp.clip(e_s / (params["stoch_dedx"] * ell), 1.0, float(n))
    w = jnp.clip(n_f[:, None] - jnp.arange(n, dtype=dt)[None, :], 0.0, 1.0)     # sum w = n_f
    edep = e_s[:, None] * w / n_f[:, None]
    tc = p.time[:, None] + t * x0 / C_LIGHT
    use = prof.use
    return Segments(p0, p1, jnp.where(use[:, None], edep, 0.0), tc - 0.5 * ell / C_LIGHT, tc + 0.5 * ell / C_LIGHT,
                    jnp.full((N, n), TRACK, jnp.int32), use[:, None] & (w > 0))


def _flatten(*segs: Segments) -> Segments:
    def cat(*xs):
        return jnp.concatenate([x.reshape((-1,) + x.shape[2:]) for x in xs], axis=0)

    return Segments(*(cat(*parts) for parts in zip(*segs)))


def from_particles(det: Detector, params, tables, p: Particles) -> Segments:
    """All segments of one event's mean deposits, flattened to ``[S]``."""
    return _flatten(track_segments(det, params, tables, p), shower_segments(det, params, p))


def sampled_from_particles(det: Detector, params, tables, p: Particles, key) -> Segments:
    """One draw of the stochastic deposit tier (``Settings.n_stoch > 0``): each shower's maximum shifted by
    ``stoch_tmax_sigma`` X0 times a standard normal, the fraction ``stoch_frac`` of its energy in charged-track pieces
    (:func:`stochastic_segments`) and the rest in the mean shower. Tracks are unchanged."""
    k_t, k_s = jax.random.split(key)
    noise = params["stoch_tmax_sigma"] * jax.random.normal(k_t, p.pdg.shape, p.vertex.dtype)
    prof = shower_profile(det, params, p, noise)
    smooth = shower_segments(det, params, p, prof, scale=jnp.broadcast_to(1.0 - params["stoch_frac"], p.pdg.shape))
    return _flatten(track_segments(det, params, tables, p), smooth, stochastic_segments(det, params, p, prof, key=k_s))
