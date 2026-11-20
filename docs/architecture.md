# Architecture (v0)

```text
nova-fastsim                 mu2e-fastsim / bdnmc-dune
  profiles, ND response, examples, site.env
  private: nova-fastsim-inputs (follow-up)
        │
hep-simlib / hep-simbench / hep-endpoints / cryjax
        │
BdNMC + ldmgen + bdnmc-nova (+ g4numi-ldm-tools)   ← not merged here
```

Event record (agreed): keep `ldmgen/1` HDF5 for analysis; public = HepMC3 + LdmHepMC;
HEPEVT / TextFileGen / edep-sim / Mu2e GenParticle are sinks only.

Plan SoT: private workspace wiki `<private-nova-wiki>/sim/nova-fastsim-plan`.
