# nova-fastsim

JAX-oriented NOvA Near Detector simulation experiment package (parallel to
[`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim)). Profiles, ND response,
examples and site configuration live here. Generators (`BdNMC`, `ldmgen`,
[`bdnmc-nova`](https://github.com/RobMina/bdnmc-nova)) and hep-sim\* stay in their own
repositories — this package **depends on** them and does not merge them.

Private geometry / FCLs belong in **`nova-fastsim-inputs`** (not cut yet; see
[`setup/README.md`](setup/README.md)). Code only: no NOvA-internal conditions, tables
or results.

## Status

| Piece | State |
|---|---|
| `core/` profile + `NOVA_DEVICE_FLOAT` | present; synthetic pytest |
| `ldm/` thin orchestration | **phase 2**: `python -m novafastsim_ldm` → bdnmc-nova generate/submit; optional ldmgen bridge |
| `calib/`, `beamline/` | package stubs |
| `examples/` | CLI + UVA two-stage docs |
| `benchmarks/hepsimbench_nd.py` | hep-simbench stub (status / synthetic) |
| `container/nova-fastsim-jax.def` | stub (JAX host — **not** SL7; no SIF build yet) |
| `nova-fastsim-inputs` | documented; first release not cut |
| On-device ND response | later phase |

Canonical plan (ai-config wiki): `nova/wiki/sim/nova-fastsim-plan.md`.
Shared event record: `nova/wiki/bdnmc/shared-event-record.md`.

## Layout

```text
core/          novafastsim — profile, dtype, site helpers
calib/         novafastsim_calib — hep-simlib banks under the nova profile (stub)
ldm/           novafastsim_ldm — thin BdNMC / ldmgen / bdnmc-nova orchestration
beamline/      novafastsim_beamline — flux hooks (stub)
examples/      LDM chain CLI docs
benchmarks/    hep-simbench ND stub
setup/         nova-fastsim-inputs release notes + RELEASE.example.json
container/     Apptainer def for the JAX host (two-image with NOvASoft SL7)
tests/         synthetic pytest (no private inputs)
site.env.example
```

## TextFileGen (do not rewrite)

**TextFileGen** is the stock NOvASoft art module (HEPEVT text → MCTruth → G4). This
repository does not reimplement it. `bdnmc_nova.textfilegen` / `ldmgen export nova`
**write** the HEPEVT file; NOvASoft consumes it. See
[`examples/README.md`](examples/README.md).

Two-image composition remains mandatory: JAX needs glibc ≥ 2.27; NOvASoft runs in SL7.

## Device / dtype

Every public path will support **float32 and float64**. Set `NOVA_DEVICE_FLOAT` to
`float32` or `float64`; unset defaults to `float64`. Other values fail closed.

## Quickstart

```bash
git clone https://github.com/RobMina/nova-fastsim.git && cd nova-fastsim
python3 -m venv .venv && .venv/bin/pip install -q -e core -e calib -e ldm -e beamline
.venv/bin/python -m pytest -q tests
.venv/bin/python -m novafastsim_ldm describe
```

For generation / submit against a real site:

```bash
.venv/bin/pip install -q -e 'ldm[chains]'   # bdnmc-nova + bdnmc-jax
cp site.env.example site.env   # fill paths
.venv/bin/python -m novafastsim_ldm generate --trials 1024
.venv/bin/python -m novafastsim_ldm ldmgen-status   # public ldmgen not required yet
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
