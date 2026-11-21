"""Primary particles in the detector frame: cm, ns, MeV.

``Particles`` is a pytree of fixed-shape arrays (padding rows have ``mask`` False), so a batch of events is the same
structure with a leading event axis and goes through ``jax.vmap``. Sources: arrays already on the device
(:func:`from_bdnmc`, the generator's accepted rows without a host round trip) or HEPEVT text (:func:`from_hepevt`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# |PDG| -> (mass MeV, model): model 1 = electromagnetic shower, 2 = charged track; others deposit nothing
MASS = {11: 0.51099895, 22: 0.0, 13: 105.6583755, 211: 139.57039, 321: 493.677, 2212: 938.27208816}
TRACK_SPECIES = (13, 211, 321, 2212)


class Particles(NamedTuple):
    pdg: object       # int32 [N]
    vertex: object    # [N, 3] cm
    momentum: object  # [N, 3] MeV / c
    time: object      # [N] ns
    mask: object      # bool [N]


def make(pdg, vertex, momentum, time=None, mask=None, dtype=None) -> Particles:
    """Particles from arrays (momenta in MeV / c); ``dtype`` defaults to the vertex dtype."""
    vertex = jnp.asarray(vertex, dtype)
    dt = vertex.dtype
    pdg = jnp.asarray(pdg, jnp.int32)
    momentum = jnp.asarray(momentum, dt)
    time = jnp.zeros(pdg.shape, dt) if time is None else jnp.asarray(time, dt)
    mask = jnp.ones(pdg.shape, bool) if mask is None else jnp.asarray(mask, bool)
    return Particles(pdg, vertex, momentum, time, mask)


def model_of(pdg):
    """Per particle: 1 shower (e+-, photon), 2 track (mu, pi, K, p), 0 none."""
    a = jnp.abs(pdg)
    shower = (a == 11) | (a == 22)
    track = jnp.zeros(a.shape, bool)
    for s in TRACK_SPECIES:
        track = track | (a == s)
    return jnp.where(shower, 1, jnp.where(track, 2, 0))


def species_index(pdg):
    """Index into ``TRACK_SPECIES`` (0 for non-tracks)."""
    a = jnp.abs(pdg)
    idx = jnp.zeros(a.shape, jnp.int32)
    for i, s in enumerate(TRACK_SPECIES):
        idx = jnp.where(a == s, i, idx)
    return idx


def mass_of(pdg, dtype):
    a = jnp.abs(pdg)
    m = jnp.zeros(a.shape, dtype)
    for k, v in MASS.items():
        m = jnp.where(a == k, v, m)
    return m


def from_bdnmc(electron, vertex, origin_m=(0.0, 0.0, 0.0), rotation=None, mask=None) -> Particles:
    """Recoil electrons from generator rows: ``electron [N, 4]`` (E, px, py, pz in GeV) and ``vertex [N, 4]``
    (x, y, z in m and t in s, beam frame). Detector frame: ``R (x - origin)``. Pure jnp: stays on the device."""
    electron = jnp.asarray(electron)
    vertex = jnp.asarray(vertex, electron.dtype)
    R = jnp.eye(3, dtype=electron.dtype) if rotation is None else jnp.asarray(rotation, electron.dtype)
    pos = (vertex[:, :3] - jnp.asarray(origin_m, electron.dtype)) @ R.T * 100.0
    mom = electron[:, 1:] @ R.T * 1000.0
    n = electron.shape[0]
    return make(jnp.full(n, 11, jnp.int32), pos, mom, vertex[:, 3] * 1e9, mask)


def from_hepevt(text: str, *, max_particles: int | None = None, dtype=np.float64) -> Particles:
    """A batch ``[events, max_particles]`` from HEPEVT text (``status pdg m1 m2 d1 d2 px py pz E m x y z t``, GeV,
    cm, ns). Only status-1 particles are kept; events are padded to the largest (or ``max_particles``)."""
    rows = []
    lines = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        head = lines[i]
        if len(head) != 2:
            raise ValueError(f"expected an event header '<event> <particles>', got {' '.join(head)!r}")
        n = int(head[1])
        ev = [[float(v) for v in f] for f in lines[i + 1:i + 1 + n]]
        if len(ev) != n or any(len(f) != 15 for f in ev):
            raise ValueError(f"event {head[0]}: expected {n} particle lines of 15 fields")
        rows.append([f for f in ev if int(f[0]) == 1])
        i += 1 + n
    n_max = max((len(r) for r in rows), default=0) if max_particles is None else max_particles
    if any(len(r) > n_max for r in rows):
        raise ValueError(f"an event has more than max_particles={n_max} tracked particles")
    E = len(rows)
    pdg = np.zeros((E, n_max), np.int32)
    vtx = np.zeros((E, n_max, 3), dtype)
    mom = np.zeros((E, n_max, 3), dtype)
    t = np.zeros((E, n_max), dtype)
    mask = np.zeros((E, n_max), bool)
    for e, r in enumerate(rows):
        for k, f in enumerate(r):
            pdg[e, k] = int(f[1])
            mom[e, k] = np.array(f[6:9]) * 1000.0
            vtx[e, k] = f[11:14]
            t[e, k] = f[14]
            mask[e, k] = True
    return make(pdg, vtx, mom, t, mask, dtype)
