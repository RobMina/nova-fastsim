# nova-fastsim

JAX fast simulation package for the NOvA Near Detector. The goal is a fully on-device,
fully differentiable detector simulation: particles to read-out cells to event-level
quantities on the accelerator, with gradients in the response parameters and the
particle kinematics. Profiles, the detector simulation, calibration fits, examples and
site configuration live here. Generators (`BdNMC`, `ldmgen`,
[`bdnmc-nova`](https://github.com/RobMina/bdnmc-nova)) and the hep-sim\* libraries
stay in their own repositories: this package **depends on** them and does not merge
them.

Code only: no collaboration-internal geometry, configuration, conditions, tables or
results. Those belong in a separate private inputs bundle that this package can
verify but does not contain (see [`setup/README.md`](setup/README.md)).

## Status

| Piece | State |
|---|---|
| `core/` profile + `NOVA_DEVICE_FLOAT` | present; synthetic pytest |
| `ldm/` thin orchestration | **phase 2+**: `novafastsim_ldm` → bdnmc-nova; [`ldmgen`](https://github.com/RobMina/ldmgen) library bridge (`[ldmgen]` extra); `check-site` + cuboid-override gate |
| `core/` inputs | `python -m novafastsim.inputs` `validate` / `check` for private-inputs manifests |
| `calib/` | scaffolding: `novafastsim_calib` bank layout + optional fibre probe |
| `detsim/` on-device detector simulation | **v0**: `novafastsim_detsim` — mean track + EM-shower deposits, plane/cell geometry, reco E and θ², calibration fits through hep-simlib; cell optics and APD/DCS readout from [scint-fibre-fastsim](https://github.com/RobMina/scint-fibre-fastsim) (expected + sampled); f32/f64; autodiff checked against finite differences ([`docs/detsim.md`](docs/detsim.md)) |
| `beamline/` | package stub |
| `examples/` | CLI, site smoke and two-stage docs; generic cuboid reference card |
| `ldm/` HEPEVT helpers | `events-info` / `events-head` (validate the HEPEVT text handed to detector simulation) |
| `benchmarks/hepsimbench_nd.py` | hep-simbench stub (`novafastsim[bench]`) |
| `container/nova-fastsim-jax.def` | stub (JAX host image; no SIF build yet) |
| Private inputs bundle | schema checker + role-based defaults; stub release only (identity frame, needs `NOVA_FASTSIM_ALLOW_STUB=1`) |
| Validation against the stock simulation | next: hep-simbench workflow on stock ND samples |

## Layout

```text
core/          novafastsim — profile, dtype, site helpers, inputs-manifest checker
detsim/        novafastsim_detsim — on-device differentiable detector simulation, reco, calibration fits
calib/         novafastsim_calib — bank layout scaffolding + optional fibre probe
ldm/           novafastsim_ldm — thin BdNMC / ldmgen / bdnmc-nova orchestration + HEPEVT helpers
beamline/      novafastsim_beamline — flux hooks (stub)
examples/      LDM chain CLI docs
benchmarks/    hep-simbench stub
setup/         private-inputs release notes + RELEASE.example.json
container/     Apptainer def for the JAX host
tests/         synthetic pytest (no private inputs)
site.env.example
```

## Detector simulation on the device

`detsim/` is the detector simulation: it takes generator output directly on the device
(`from_bdnmc`, no file) or from HEPEVT text (`from_hepevt`), and everything downstream
is JAX. The expected response is deterministic and differentiable; the sampled readout
draws around it. See [`docs/detsim.md`](docs/detsim.md) for the models, the detector
description (public caricature by default; real values from the private inputs bundle),
the validation table and the limitations.

The stock collaboration simulation is the **validation reference** and stays the
production path until the on-device simulation passes its fast-vs-stock benchmarks.
`novafastsim_ldm submit` runs it unchanged in its own image (two images: JAX needs
glibc ≥ 2.27, the stock simulation image is older). See
[`examples/README.md`](examples/README.md).

## Device / dtype

Every public path will support **float32 and float64**. Set `NOVA_DEVICE_FLOAT` to
`float32` or `float64`; unset defaults to `float64`. Other values fail closed.
`novafastsim.dtype.resolve()` reads the environment lazily (after `site.env`);
`submit` / `generate` always apply that resolved precision. Use
`dtype.apply_jax_x64()` so JAX’s `jax_enable_x64` matches.

## Quickstart

```bash
git clone https://github.com/RobMina/nova-fastsim.git && cd nova-fastsim
python3 -m venv .venv && .venv/bin/pip install -q -e 'core[test]' -e calib -e ldm -e beamline -e detsim
.venv/bin/python -m pytest -q
.venv/bin/python -m novafastsim_ldm describe
.venv/bin/python -m novafastsim_calib describe
.venv/bin/python -m novafastsim_detsim describe
.venv/bin/python -m novafastsim_detsim bench --events 64
```

For generation / submit against a real site:

```bash
.venv/bin/pip install -q -e 'ldm[chains,ldmgen]'   # bdnmc-nova + bdnmc-jax + ldmgen
cp site.env.example site.env   # fill paths
.venv/bin/python -m novafastsim_ldm generate --trials 1024
.venv/bin/python -m novafastsim_ldm events-info events.txt   # validate HEPEVT before sim
.venv/bin/python -m novafastsim_ldm ldmgen-status
.venv/bin/python -m novafastsim_calib status
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
