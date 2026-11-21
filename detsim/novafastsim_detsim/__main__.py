"""``python -m novafastsim_detsim describe | bench``."""

from __future__ import annotations

import argparse
import json
import time


def _setup(dtype):
    from novafastsim import dtype as nd

    name = nd.apply_jax_x64(dtype)
    return name


def describe(args) -> int:
    from dataclasses import asdict

    from . import config

    det, params = config.load(args.config) if args.config else (config.Detector(), config.default_params())
    out = dict(geometry=asdict(det.geometry), settings=asdict(det.settings),
               derived=dict(cell_pitch=det.geometry.cell_pitch, plane_pitch=det.geometry.plane_pitch,
                            cell_length=det.geometry.cell_length, z_back=det.geometry.z_back,
                            scint_fraction=det.geometry.scint_fraction, channels=det.n_channels,
                            plane_window=det.plane_window, cell_window=det.cell_window),
               params={k: float(v) for k, v in params.items() if k != "channel_eff"})
    print(json.dumps(out, indent=2))
    return 0


def bench(args) -> int:
    name = _setup(args.dtype)
    import jax
    import jax.numpy as jnp
    import numpy as np

    from . import config, reco, sim
    from .particles import make

    det, params = config.load(args.config) if args.config else (config.Detector(), config.default_params())
    params = config.to_device(params, name)
    rng = np.random.default_rng(args.seed)
    n = args.events
    g = det.geometry
    vtx = np.c_[rng.uniform(-0.3, 0.3, (n, 2)) * g.cell_length, rng.uniform(0.1, 0.4, n) * g.z_back][:, None, :]
    th = np.arccos(rng.uniform(np.cos(args.max_angle), 1.0, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    E = rng.uniform(args.emin, args.emax, n)
    mom = (E[:, None] * np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])[:, None, :]
    ev = make(np.full((n, 1), 11), vtx, mom, dtype=jnp.dtype(name))

    def run(key):
        r, h = sim.sampled_batch(det, params, ev, key)
        q = jax.vmap(lambda hh: reco.readout_charge(hits=hh, params=params))(h)
        e = jax.vmap(lambda qq: reco.energy(det, params, qq))(q)
        return h.hit.sum(), e

    f = jax.jit(run)
    t0 = time.perf_counter()
    jax.block_until_ready(f(jax.random.PRNGKey(0)))
    t1 = time.perf_counter()
    for i in range(args.repeat):
        jax.block_until_ready(f(jax.random.PRNGKey(i + 1)))
    t2 = time.perf_counter()
    rate = n * args.repeat / (t2 - t1)
    print(json.dumps(dict(device=str(jax.devices()[0]), dtype=name, events=n, compile_s=round(t1 - t0, 3),
                          warm_events_per_s=round(rate, 1), channels=det.n_channels)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="novafastsim_detsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="print the detector description (defaults or --config)")
    d.add_argument("--config")
    d.set_defaults(fn=describe)
    b = sub.add_parser("bench", help="warm events/s of sampled electrons -> hits -> calorimetric energy")
    b.add_argument("--config")
    b.add_argument("--events", type=int, default=64)
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("--emin", type=float, default=200.0)
    b.add_argument("--emax", type=float, default=3000.0)
    b.add_argument("--max-angle", type=float, default=0.3)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--dtype", choices=("float32", "float64"))
    b.set_defaults(fn=bench)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
