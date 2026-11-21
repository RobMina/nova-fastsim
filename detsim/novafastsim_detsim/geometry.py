"""Straight segments to scintillator path lengths per cell, on device and differentiable in the endpoints.

Each segment is intersected with a window of ``plane_window`` planes starting at the plane of its lowest z, and in
each plane with ``cell_window`` cells starting at the cell of its lowest transverse coordinate. Segments no longer
than ``Settings.max_segment`` always fit their windows; longer ones are flagged in ``Pieces.overflow``. The path
length in a cell is the overlap of three parameter intervals (plane z, cell u, cell length w), a piecewise-linear
function of the endpoints.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Detector

_BIG = 1e30


class Pieces(NamedTuple):
    """Segment-cell overlaps, ``[S, K]`` with ``K = plane_window * cell_window``."""

    index: object      # flat channel index plane * n_cells + cell (clipped; length 0 where invalid)
    length: object     # scintillator path length, cm
    t_mid: object      # segment parameter at the middle of the overlap
    distance: object   # distance along the cell from the overlap middle to the readout end, cm
    len_box: object    # [S] path length inside the detector envelope, cm
    overflow: object   # [S] True if the segment did not fit its windows


def slab(x0, dx, lo, hi):
    """Parameter interval ``(t0, t1)`` where ``x0 + t dx`` lies in ``[lo, hi]``; empty when ``t0 > t1``."""
    small = jnp.abs(dx) < 1e-12
    safe = jnp.where(small, 1.0, dx)
    ta = (lo - x0) / safe
    tb = (hi - x0) / safe
    inside = (x0 >= lo) & (x0 <= hi)
    t0 = jnp.where(small, jnp.where(inside, -_BIG, _BIG), jnp.minimum(ta, tb))
    t1 = jnp.where(small, jnp.where(inside, _BIG, -_BIG), jnp.maximum(ta, tb))
    return t0, t1


def box_interval(det: Detector, p0, d):
    """Parameter interval of ``p0 + t d`` inside the detector envelope (no clipping to [0, 1])."""
    g = det.geometry
    h = g.half_width
    x0, x1 = slab(p0[..., 0], d[..., 0], -h, h)
    y0, y1 = slab(p0[..., 1], d[..., 1], -h, h)
    z0, z1 = slab(p0[..., 2], d[..., 2], g.z_front, g.z_back)
    return jnp.maximum(jnp.maximum(x0, y0), z0), jnp.minimum(jnp.minimum(x1, y1), z1)


def _norm(d):
    sq = jnp.sum(d * d, axis=-1)
    pos = sq > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, sq, 1.0)), 0.0)


def pieces(det: Detector, p0, p1) -> Pieces:
    """Overlaps of segments ``p0 -> p1`` (``[S, 3]``, cm) with the cells."""
    g = det.geometry
    P, C, Kp, Kc = g.n_planes, g.n_cells, det.plane_window, det.cell_window
    h, pp, cp = g.half_width, g.plane_pitch, g.cell_pitch
    d = p1 - p0
    L = _norm(d)

    b0, b1 = box_interval(det, p0, d)
    len_box = jnp.maximum(jnp.minimum(b1, 1.0) - jnp.maximum(b0, 0.0), 0.0) * L

    z_a, z_b = jax.lax.stop_gradient(jnp.minimum(p0[:, 2], p1[:, 2])), jax.lax.stop_gradient(
        jnp.maximum(p0[:, 2], p1[:, 2]))
    p_first = jnp.floor((z_a - g.z_front) / pp).astype(jnp.int32)
    p_last = jnp.floor((z_b - g.z_front) / pp).astype(jnp.int32)
    p_start = jnp.clip(p_first, 0, P - 1)
    planes = p_start[:, None] + jnp.arange(Kp, dtype=jnp.int32)          # [S, Kp]
    valid_p = planes < P
    ovf_p = (jnp.minimum(p_last, P - 1) - jnp.maximum(p_first, 0) + 1) > Kp

    z_lo = g.z_front + planes * pp + g.face
    tz0, tz1 = slab(p0[:, None, 2], d[:, None, 2], z_lo, z_lo + g.cell_depth)
    view = (planes + g.first_view) % 2
    is_x = view == 0
    u0 = jnp.where(is_x, p0[:, None, 0], p0[:, None, 1])
    du = jnp.where(is_x, d[:, None, 0], d[:, None, 1])
    w0 = jnp.where(is_x, p0[:, None, 1], p0[:, None, 0])
    dw = jnp.where(is_x, d[:, None, 1], d[:, None, 0])
    tw0, tw1 = slab(w0, dw, -h, h)
    ta = jnp.maximum(jnp.maximum(tz0, tw0), 0.0)
    tb = jnp.minimum(jnp.minimum(tz1, tw1), 1.0)
    hit_p = (ta < tb) & valid_p

    ua = jax.lax.stop_gradient(u0 + jnp.clip(ta, 0.0, 1.0) * du)
    ub = jax.lax.stop_gradient(u0 + jnp.clip(tb, 0.0, 1.0) * du)
    c_first = jnp.floor((jnp.minimum(ua, ub) + h) / cp).astype(jnp.int32)
    c_last = jnp.floor((jnp.maximum(ua, ub) + h) / cp).astype(jnp.int32)
    c_start = jnp.clip(c_first, 0, C - 1)
    ovf_c = jnp.any(hit_p & ((jnp.minimum(c_last, C - 1) - jnp.maximum(c_first, 0) + 1) > Kc), axis=1)
    cells = c_start[..., None] + jnp.arange(Kc, dtype=jnp.int32)          # [S, Kp, Kc]
    valid = hit_p[..., None] & (cells < C)

    u_lo = -h + cells * cp + 0.5 * g.web
    tu0, tu1 = slab(u0[..., None], du[..., None], u_lo, u_lo + g.cell_width)
    t0 = jnp.maximum(ta[..., None], tu0)
    t1 = jnp.minimum(tb[..., None], tu1)
    length = jnp.where(valid, jnp.maximum(t1 - t0, 0.0), 0.0) * L[:, None, None]
    t_mid = jnp.where(length > 0, 0.5 * (t0 + t1), 0.0)
    w_mid = w0[..., None] + t_mid * dw[..., None]
    sign = jnp.where(is_x, g.readout_end[0], g.readout_end[1])[..., None]
    distance = jnp.clip(h - sign * w_mid, 0.0, 2.0 * h)

    index = jnp.clip(planes, 0, P - 1)[..., None] * C + jnp.clip(cells, 0, C - 1)
    S = p0.shape[0]
    flat = lambda a: a.reshape(S, Kp * Kc)
    return Pieces(flat(index), flat(length), flat(t_mid), flat(distance), len_box, ovf_p | ovf_c)


def cell_centres(det: Detector):
    """``(u [C], z [P], view [P])``: transverse cell centres, plane centres along z and plane views."""
    g = det.geometry
    u = -g.half_width + (jnp.arange(g.n_cells) + 0.5) * g.cell_pitch
    p = jnp.arange(g.n_planes)
    z = g.z_front + p * g.plane_pitch + g.face + 0.5 * g.cell_depth
    return u, z, (p + g.first_view) % 2
