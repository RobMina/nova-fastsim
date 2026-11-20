# nova-fastsim

JAX-oriented NOvA Near Detector simulation experiment package (parallel to
[`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim)). Profiles, ND response,
examples and site configuration live here. Generators (`BdNMC`, `ldmgen`,
[`bdnmc-nova`](https://github.com/RobMina/bdnmc-nova)) and hep-sim\* stay in their own
repositories — this package **depends on** them and does not merge them.

Private geometry / FCLs belong in a separate **`nova-fastsim-inputs`** bundle (not
created in this v0). Code only: no NOvA-internal conditions, tables or results.

## Status (v0)

Scaffold only (plan phase 1 / milestones S1–S2, thin S3 docs):

| Piece | State |
|---|---|
| `core/` profile + `NOVA_DEVICE_FLOAT` | present; synthetic pytest |
| `calib/`, `ldm/`, `beamline/` | package stubs |
| `examples/` | points at the existing bdnmc-nova / UVA two-stage chain |
| `container/nova-fastsim-jax.def` | stub (JAX host image — **not** SL7; no SIF build yet) |
| `nova-fastsim-inputs` | follow-up |
| On-device ND response | later phase |

Canonical plan (ai-config wiki): `nova/wiki/sim/nova-fastsim-plan.md`.
Shared event record: `nova/wiki/bdnmc/shared-event-record.md`.

## Layout

```text
core/          novafastsim — profile, dtype, site helpers
calib/         novafastsim_calib — hep-simlib banks under the nova profile (stub)
ldm/           novafastsim_ldm — thin orchestration over BdNMC / ldmgen / bdnmc-nova (stub)
beamline/      novafastsim_beamline — flux hooks (stub)
examples/      documented entry points into the existing UVA chain
benchmarks/    placeholders
container/     Apptainer def for the JAX host (two-image composition with NOvASoft SL7)
tests/         synthetic pytest (no private inputs)
site.env.example
```

## TextFileGen (do not rewrite)

**TextFileGen** is the stock NOvASoft art module (HEPEVT text → MCTruth → G4). This
repository does not reimplement it. `bdnmc_nova.textfilegen` / `ldmgen export nova`
**write** the HEPEVT file; NOvASoft consumes it. See
[`examples/README.md`](examples/README.md) and
[`bdnmc-nova`](https://github.com/RobMina/bdnmc-nova) (`jobs/two_stage.sh`).

Two-image composition remains mandatory: JAX needs glibc ≥ 2.27; NOvASoft runs in SL7.

## Device / dtype

Every public path will support **float32 and float64**. Set `NOVA_DEVICE_FLOAT` to
`float32` or `float64`; unset defaults to `float64`. Other values fail closed.

## Quickstart (v0)

```bash
git clone https://github.com/RobMina/nova-fastsim.git && cd nova-fastsim
python3 -m venv .venv && .venv/bin/pip install -q -e core -e calib -e ldm -e beamline
.venv/bin/python -m pytest -q tests
```

No private inputs are required for the synthetic tests. Running the LDM example chain
needs a site `site.env`, the bdnmc-nova checkout, and NOvASoft / Apptainer on the
target host (UVA or FNAL) — see `examples/README.md`.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
