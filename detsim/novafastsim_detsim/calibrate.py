"""Calibration fits of simulation parameters to per-cell counts, through ``hepsimlib.fit``.

The expected response is the model; Jacobians come from ``jax.jacfwd``; the optimizer, covariance and conventions
are hep-simlib's (Fisher scoring on a Poisson likelihood, float64 host algebra).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .config import Detector
from .sim import expected_batch


@dataclass
class CalibrationFit:
    names: tuple
    theta: np.ndarray
    errors: np.ndarray
    cov: np.ndarray
    result: object    # hepsimlib.fit.FitResult


def expected_counts(det: Detector, params, particles, names, observable="pe"):
    """``theta -> per-cell expected counts [events * channels]`` with ``names`` replaced by ``theta``."""
    base = dict(params)

    def fn(theta):
        pr = dict(base)
        pr.update({n: theta[i] for i, n in enumerate(names)})
        return getattr(expected_batch(det, pr, particles), observable).reshape(-1)

    return fn


def fit_poisson(det: Detector, params, particles, observed, names, *, min_expected=0.5, observable="pe",
                **gn_kw) -> CalibrationFit:
    """Fit ``names`` to observed per-cell counts ``[events, planes, cells]`` (Poisson). Cells whose expected count
    at the start values is below ``min_expected`` are left out."""
    from hepsimlib import fit

    fn = expected_counts(det, params, particles, names, observable)
    theta0 = np.array([float(params[n]) for n in names])
    dt = params[names[0]].dtype
    mu0 = np.asarray(fn(jnp.asarray(theta0, dt)))
    sel = np.flatnonzero(mu0 >= min_expected)
    obs = np.asarray(observed).reshape(-1)[sel]
    obj = fit.poisson(lambda th: fn(th)[sel], obs, jac="fwd", dtype=np.dtype(dt))
    res = fit.gauss_newton(obj, theta0, **{"scale": "auto", **gn_kw})
    cov = fit.covariance(res.info)
    return CalibrationFit(tuple(names), np.asarray(res.theta), np.sqrt(np.diag(cov)), cov, res)
