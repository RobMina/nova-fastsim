# Architecture

```text
nova-fastsim                 mu2e-fastsim / bdnmc-dune
  profiles, ND response, examples, site.env
  private: nova-fastsim-inputs (documented; first release TBD)
        │
hep-simlib / hep-simbench / hep-endpoints / cryjax
        │
BdNMC + ldmgen + bdnmc-nova (+ g4numi-ldm-tools)   ← not merged here
```

Phase-2 LDM path (orchestrate only):

```text
novafastsim_ldm generate  →  bdnmc_nova.generate (BdNMC JAX → events.txt)
        │ optional
        ▼
ldmgen convert / export   →  RobMina/ldmgen; still sinks through bdnmc_nova.textfilegen
        │
novafastsim_ldm events-*  →  validate / head HEPEVT TextFileGen *input* text
        │
novafastsim_ldm submit    →  bdnmc_nova.jobspec → stock NOvASoft TextFileGen (SL7)
```

Phase-2 calib path (parallel; scaffolding):

```text
novafastsim_calib describe/status/ensure-dirs
        → nova profile bank layout under NOVA_LIBRARY_CACHE
        → optional scint-fibre-fastsim probe (NOvA APD stub lives there)
```

Event record (agreed): keep `ldmgen/1` HDF5 for analysis; public = HepMC3 + LdmHepMC;
HEPEVT / TextFileGen / edep-sim / Mu2e GenParticle are sinks only.

Plan SoT: private workspace wiki `<private-nova-wiki>/sim/nova-fastsim-plan`.
Inputs: [`../setup/README.md`](../setup/README.md). hep-simbench stub:
[`../benchmarks/README.md`](../benchmarks/README.md).
