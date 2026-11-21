# novafastsim-detsim

On-device, differentiable Near Detector simulation in JAX: particles → energy-deposit segments → cell path
lengths → cell response → per-cell expected response and a sampled readout → calorimetric energy and direction.
float32 and float64; `jit` per event, `vmap` per batch.

The cell response is [scint-fibre-fastsim](https://github.com/RobMina/scint-fibre-fastsim)'s: its NOvA cell
(`geometry.nova_cell_tdr`), cell optics (`response.light`: Birks quenching, looped-fibre transport, analytic or a
`CellTable`) and NOvA APD + front-end backend (`backends.nova_apd`). This package owns the deposits, the plane/cell
geometry, reco and calibration fits; parameters `optics.*` and `apd.*` are scint-fibre-fastsim's fields and units.

```python
import jax, jax.numpy as jnp
from novafastsim import dtype
dtype.apply_jax_x64("float64")                        # NOVA_DEVICE_FLOAT when called without an argument
import novafastsim_detsim as ds
from novafastsim_detsim import reco

det = ds.Detector()                                   # public caricature; ds.load(path) for real inputs
params = ds.to_device(ds.default_params(), "float64")
ev = ds.make([11], [[0.0, 0.0, 300.0]], [[50.0, 0.0, 1500.0]])    # cm, MeV/c
r = ds.expected(det, params, ev)                      # Response: edep, evis, pe, time, adc, p_hit, diag
r, hits = ds.sampled(det, params, ev, jax.random.PRNGKey(0))
E = reco.energy(det, params, reco.readout_charge(r))
k = "optics.yield_per_mev"                            # scint-fibre-fastsim CellOptics field
dE = jax.grad(lambda y: reco.energy(det, {**params, k: y}, reco.readout_charge(
    ds.expected(det, {**params, k: y}, ev))))(params[k])
```

```bash
python -m novafastsim_detsim describe [--config detector.json]
python -m novafastsim_detsim bench --events 64 --dtype float32
```

Design, models, parameters, validation and limitations: [`../docs/detsim.md`](../docs/detsim.md).
