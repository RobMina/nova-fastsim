"""Straight segments to scintillator path lengths per cell, on device and differentiable in the endpoints.

Each segment is intersected with a window of ``plane_window`` planes starting at the plane of its lowest z, and in
each plane with ``cell_window`` cells starting at the cell of its lowest transverse coordinate. Segments no longer
than ``Settings.max_segment`` always fit their windows; longer ones are flagged in ``Pieces.overflow``. The path
length in a cell is the overlap of three parameter intervals (plane z, cell u, cell length w), a piecewise-linear
function of the endpoints. A uniform geometry finds the windows by division; a tabulated one (``Geometry.plane_z``
etc.) by counting table edges, with every cell's own centre and width and every plane's own z, view and long-axis
centre.
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


class Layout(NamedTuple):
    """Device arrays of a geometry: plane z of the scintillator front ``[P]``, plane view ``[P]``, long-axis centre
    ``[P]``, cell lower edges ``[P, C]`` and widths ``[P, C]`` (transverse), and the plane / cell boundaries used to
    find the windows (``[P + 1]`` and ``[P, C + 1]``)."""

    z_lo: object
    view: object
    w_centre: object
    u_lo: object
    u_width: object
    plane_edges: object
    cell_edges: object


def layout(det: Detector, dtype=jnp.float32) -> Layout:
    g = det.geometry
    P, C = g.n_planes, g.n_cells
    if g.tabulated:
        pz = jnp.asarray(g.plane_z, dtype)
        cu = jnp.asarray(g.cell_u, dtype)
        cw = jnp.asarray(g.cell_w, dtype)
        view = jnp.asarray(g.plane_view, jnp.int32)
        wc = jnp.asarray(g.long_centre, dtype)
        mid_p = 0.5 * (pz[1:] + pz[:-1])
        pe = jnp.concatenate([pz[:1] - 0.5 * g.plane_pitch, mid_p, pz[-1:] + 0.5 * g.plane_pitch])
        mid_c = 0.5 * (cu[:, 1:] + cu[:, :-1])
        ce = jnp.concatenate([cu[:, :1] - 0.5 * cw[:, :1] - 0.5 * g.web, mid_c,
                              cu[:, -1:] + 0.5 * cw[:, -1:] + 0.5 * g.web], axis=1)
        return Layout(pz - 0.5 * g.cell_depth, view, wc, cu - 0.5 * cw, cw, pe, ce)
    p = jnp.arange(P)
    pp, cp, h = g.plane_pitch, g.cell_pitch, g.half_width
    z_lo = (g.z_front + p * pp + g.face).astype(dtype)
    u_lo = jnp.broadcast_to((-h + jnp.arange(C) * cp + 0.5 * g.web).astype(dtype), (P, C))
    pe = (g.z_front + jnp.arange(P + 1) * pp).astype(dtype)
    ce = jnp.broadcast_to((-h + jnp.arange(C + 1) * cp).astype(dtype), (P, C + 1))
    return Layout(z_lo, (p + g.first_view) % 2, jnp.zeros(P, dtype), u_lo, jnp.full((P, C), g.cell_width, dtype), pe, ce)


def box_interval(det: Detector, p0, d):
    """Parameter interval of ``p0 + t d`` inside the detector envelope (no clipping to [0, 1])."""
    g = det.geometry
    h = g.half_width
    x0, x1 = slab(p0[..., 0], d[..., 0], -h, h)
    y0, y1 = slab(p0[..., 1], d[..., 1], -h, h)
    z0, z1 = slab(p0[..., 2], d[..., 2], g.z_front_edge, g.z_back)
    return jnp.maximum(jnp.maximum(x0, y0), z0), jnp.minimum(jnp.minimum(x1, y1), z1)


def _norm(d):
    sq = jnp.sum(d * d, axis=-1)
    pos = sq > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, sq, 1.0)), 0.0)


def pieces(det: Detector, p0, p1) -> Pieces:
    """Overlaps of segments ``p0 -> p1`` (``[S, 3]``, cm) with the cells."""
    g = det.geometry
    P, C, Kp, Kc = g.n_planes, g.n_cells, det.plane_window, det.cell_window
    lay = layout(det, p0.dtype)
    half_len = 0.5 * g.cell_length
    d = p1 - p0
    L = _norm(d)

    b0, b1 = box_interval(det, p0, d)
    len_box = jnp.maximum(jnp.minimum(b1, 1.0) - jnp.maximum(b0, 0.0), 0.0) * L

    z_a, z_b = jax.lax.stop_gradient(jnp.minimum(p0[:, 2], p1[:, 2])), jax.lax.stop_gradient(
        jnp.maximum(p0[:, 2], p1[:, 2]))
    p_first = jnp.searchsorted(lay.plane_edges, z_a, side="right").astype(jnp.int32) - 1
    p_last = jnp.searchsorted(lay.plane_edges, z_b, side="right").astype(jnp.int32) - 1
    p_start = jnp.clip(p_first, 0, P - 1)
    planes = p_start[:, None] + jnp.arange(Kp, dtype=jnp.int32)          # [S, Kp]
    valid_p = planes < P
    ovf_p = (jnp.minimum(p_last, P - 1) - jnp.maximum(p_first, 0) + 1) > Kp
    pc = jnp.clip(planes, 0, P - 1)

    z_lo = lay.z_lo[pc]
    tz0, tz1 = slab(p0[:, None, 2], d[:, None, 2], z_lo, z_lo + g.cell_depth)
    view = lay.view[pc]
    is_x = view == 0
    u0 = jnp.where(is_x, p0[:, None, 0], p0[:, None, 1])
    du = jnp.where(is_x, d[:, None, 0], d[:, None, 1])
    w0 = jnp.where(is_x, p0[:, None, 1], p0[:, None, 0])
    dw = jnp.where(is_x, d[:, None, 1], d[:, None, 0])
    wc = lay.w_centre[pc]
    tw0, tw1 = slab(w0, dw, wc - half_len, wc + half_len)
    ta = jnp.maximum(jnp.maximum(tz0, tw0), 0.0)
    tb = jnp.minimum(jnp.minimum(tz1, tw1), 1.0)
    hit_p = (ta < tb) & valid_p

    ua = jax.lax.stop_gradient(u0 + jnp.clip(ta, 0.0, 1.0) * du)
    ub = jax.lax.stop_gradient(u0 + jnp.clip(tb, 0.0, 1.0) * du)
    edges = lay.cell_edges[pc][..., 1:-1]                                  # [S, Kp, C - 1] inner boundaries
    c_first = jnp.sum(edges <= jnp.minimum(ua, ub)[..., None], axis=-1).astype(jnp.int32)
    c_last = jnp.sum(edges <= jnp.maximum(ua, ub)[..., None], axis=-1).astype(jnp.int32)
    c_first = jnp.where(jnp.minimum(ua, ub) < lay.cell_edges[pc][..., 0], -1, c_first)
    c_last = jnp.where(jnp.maximum(ua, ub) >= lay.cell_edges[pc][..., -1], C, c_last)
    c_start = jnp.clip(c_first, 0, C - 1)
    ovf_c = jnp.any(hit_p & ((jnp.minimum(c_last, C - 1) - jnp.maximum(c_first, 0) + 1) > Kc), axis=1)
    cells = c_start[..., None] + jnp.arange(Kc, dtype=jnp.int32)          # [S, Kp, Kc]
    valid = hit_p[..., None] & (cells < C)

    cc = jnp.clip(cells, 0, C - 1)
    u_lo = jnp.take_along_axis(lay.u_lo[pc], cc, axis=-1)
    u_w = jnp.take_along_axis(lay.u_width[pc], cc, axis=-1)
    tu0, tu1 = slab(u0[..., None], du[..., None], u_lo, u_lo + u_w)
    t0 = jnp.maximum(ta[..., None], tu0)
    t1 = jnp.minimum(tb[..., None], tu1)
    length = jnp.where(valid, jnp.maximum(t1 - t0, 0.0), 0.0) * L[:, None, None]
    t_mid = jnp.where(length > 0, 0.5 * (t0 + t1), 0.0)
    w_mid = w0[..., None] + t_mid * dw[..., None]
    sign = jnp.where(is_x, g.readout_end[0], g.readout_end[1])[..., None]
    distance = jnp.clip(half_len - sign * (w_mid - wc[..., None]), 0.0, 2.0 * half_len)

    index = pc[..., None] * C + cc
    S = p0.shape[0]
    flat = lambda a: a.reshape(S, Kp * Kc)
    return Pieces(flat(index), flat(length), flat(t_mid), flat(distance), len_box, ovf_p | ovf_c)


def cell_centres(det: Detector, dtype=jnp.float32):
    """``(u [P, C], z [P], view [P])``: transverse cell centres, plane centres along z and plane views."""
    lay = layout(det, dtype)
    return lay.u_lo + 0.5 * lay.u_width, lay.z_lo + 0.5 * det.geometry.cell_depth, lay.view


def readout_distance(det: Detector, view, w, plane=None, dtype=jnp.float32):
    """Distance (cm) along the cell from long-axis position ``w`` to the readout end, for planes of ``view``
    (``plane`` selects each plane's own long-axis centre in a tabulated geometry)."""
    g = det.geometry
    half = 0.5 * g.cell_length
    wc = 0.0 if plane is None else layout(det, dtype).w_centre[plane]
    sign = jnp.where(view == 0, g.readout_end[0], g.readout_end[1])
    return jnp.clip(half - sign * (w - wc), 0.0, 2.0 * half)
