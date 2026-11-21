"""Mean range-energy tables of charged tracks in scintillator-equivalent column density (host, NumPy).

Bethe mean stopping power with the Sternheimer density effect for a CH2-like scintillator; below beta*gamma = 0.05
the range follows R ~ T^1.75 matched at that point. Radiative losses (muons above ~100 GeV) are not included.
"""

from __future__ import annotations

import numpy as np

from .particles import MASS, TRACK_SPECIES

K_BETHE = 0.307075   # MeV cm^2 / mol
ME = 0.51099895      # MeV
# polyethylene-like scintillator: Z/A, I (eV), Sternheimer C, x0, x1, a, k
CH2 = dict(z_over_a=0.57034, i_ev=57.4, cbar=3.0016, x0=0.1370, x1=2.5177, a=0.1211, k=3.4292)


def stopping_power(T, M, mat=CH2):
    """Mean -dE/dx (MeV cm^2 / g) of a singly charged particle of kinetic energy ``T`` and mass ``M`` (MeV)."""
    T = np.asarray(T, np.float64)
    gamma = 1.0 + T / M
    bg2 = gamma * gamma - 1.0
    b2 = bg2 / (gamma * gamma)
    r = ME / M
    tmax = 2.0 * ME * bg2 / (1.0 + 2.0 * gamma * r + r * r)
    I = mat["i_ev"] * 1e-6
    x = 0.5 * np.log10(np.maximum(bg2, 1e-30))
    ln10x = 2.0 * np.log(10.0) * x - mat["cbar"]
    delta = np.where(x >= mat["x1"], ln10x,
                     np.where(x >= mat["x0"], ln10x + mat["a"] * np.maximum(mat["x1"] - x, 0.0) ** mat["k"], 0.0))
    L = 0.5 * np.log(2.0 * ME * bg2 * tmax / (I * I)) - b2 - 0.5 * delta
    return K_BETHE * mat["z_over_a"] / b2 * L


def range_table(M, n=800, t_max=1e6, bg_low=0.05, mat=CH2):
    """``(T [n+1], R [n+1])`` with ``T[0] = R[0] = 0``: mean CSDA range in g / cm^2."""
    t_low = M * (np.sqrt(1.0 + bg_low ** 2) - 1.0)
    T = np.geomspace(t_low, t_max, n)
    S = stopping_power(T, M, mat)
    inv = 1.0 / S
    R = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(T))])
    r_low = t_low / (1.75 * S[0])         # integral of T^0.75 law matched in slope at t_low
    R = R + r_low
    T_lo = t_low * np.linspace(0.0, 1.0, 17)[:-1]
    R_lo = r_low * (T_lo / t_low) ** 1.75
    return np.concatenate([T_lo, T]), np.concatenate([R_lo, R])


def tables(n=800):
    """Stacked tables for ``particles.TRACK_SPECIES``: ``(T [4, m], R [4, m])``."""
    rows = [range_table(MASS[s], n) for s in TRACK_SPECIES]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
