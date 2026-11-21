# nova-fastsim

JAX fast simulation package for the NOvA Near Detector. Profiles, detector response,
examples and site configuration live here. Generators (`BdNMC`, `ldmgen`,
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
| `ldm/` thin orchestration | **phase 2+**: `novafastsim_ldm` → bdnmc-nova; [`ldmgen`](https://github.com/RobMina/ldmgen) library bridge (`[ldmgen]` extra) |
| `core/` inputs | `python -m novafastsim.inputs` `validate` / `check` for private-inputs manifests |
| `calib/` | scaffolding: `novafastsim_calib` bank layout + optional fibre probe |
| `beamline/` | package stub |
| `examples/` | CLI + site two-stage docs |
| `ldm/` HEPEVT helpers | `events-info` / `events-head` (validate the HEPEVT text handed to detector simulation) |
| `benchmarks/hepsimbench_nd.py` | hep-simbench stub (`novafastsim[bench]`) |
| `container/nova-fastsim-jax.def` | stub (JAX host image; no SIF build yet) |
| Private inputs bundle | documented + schema checker; first release not cut |
| On-device detector response | later phase |

## Layout

```text
core/          novafastsim — profile, dtype, site helpers, inputs-manifest checker
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

## Detector simulation stays external

Generation writes a HEPEVT text file; the detector simulation that reads it runs in
the collaboration's own software, unchanged, in its own container image. This
repository orchestrates those jobs and never reimplements that simulation. See
[`examples/README.md`](examples/README.md).

Two-image composition is mandatory: JAX needs glibc ≥ 2.27, and the detector
simulation image is older.

## Device / dtype

Every public path will support **float32 and float64**. Set `NOVA_DEVICE_FLOAT` to
`float32` or `float64`; unset defaults to `float64`. Other values fail closed.
`novafastsim.dtype.resolve()` reads the environment lazily (after `site.env`);
`submit` / `generate` always apply that resolved precision. Use
`dtype.apply_jax_x64()` so JAX’s `jax_enable_x64` matches.

## Quickstart

```bash
git clone https://github.com/RobMina/nova-fastsim.git && cd nova-fastsim
python3 -m venv .venv && .venv/bin/pip install -q -e 'core[test]' -e calib -e ldm -e beamline
.venv/bin/python -m pytest -q
.venv/bin/python -m novafastsim_ldm describe
.venv/bin/python -m novafastsim_calib describe
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
