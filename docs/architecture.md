# Architecture

```text
nova-fastsim
  profiles, on-device detector simulation (detsim), calibration, examples, site.env
  private: inputs bundle (documented; first release TBD)
        │
hep-simlib / hep-simbench / hep-endpoints / cryjax
        │
BdNMC + ldmgen + bdnmc-nova (+ g4numi-ldm-tools)   ← not merged here
```

On-device path (target):

```text
bdnmc_jax rows (device) ─┐
HEPEVT text ─────────────┼→ novafastsim_detsim.Particles
                         ▼
deposits (mean tracks + EM showers) → geometry.pieces → cell response (scint-fibre-fastsim: optics + NOvA APD)
        → expected Response (differentiable) / sampled Hits → reco (E, θ²) → fits (hepsimlib.fit)
```

Stock reference path (validation and production until the on-device simulation is validated):

```text
novafastsim_ldm generate  →  bdnmc_nova.generate (BdNMC JAX → events.txt)
        │ optional
        ▼
ldmgen convert / export   →  RobMina/ldmgen; still sinks through the bdnmc-nova HEPEVT writer
        │
novafastsim_ldm events-*  →  validate / head the HEPEVT *input* text
        │
novafastsim_ldm submit    →  bdnmc_nova.jobspec → stock detector simulation (SL7; the reference arm)
```

Phase-2 calib path (parallel; scaffolding):

```text
novafastsim_calib describe/status/ensure-dirs
        → nova profile bank layout under NOVA_LIBRARY_CACHE
        → optional scint-fibre-fastsim probe (NOvA APD stub lives there)
```

Event record (agreed): keep `ldmgen/1` HDF5 for analysis; public = HepMC3 + LdmHepMC;
HEPEVT / edep-sim / Mu2e GenParticle are sinks only.

Inputs: [`../setup/README.md`](../setup/README.md). hep-simbench stub:
[`../benchmarks/README.md`](../benchmarks/README.md).
