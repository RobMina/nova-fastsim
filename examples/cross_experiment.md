# Cross-experiment pointers (DUNE / Mu2e)

`nova-fastsim` owns NOvA ND profiles and orchestration only. Shared pieces stay outside:

| Need | Where |
|---|---|
| Event record / HepMC3 / HEPEVT sinks | [`RobMina/ldmgen`](https://github.com/RobMina/ldmgen) |
| NOvA HEPEVT writer + site jobs | [`RobMina/bdnmc-nova`](https://github.com/RobMina/bdnmc-nova) |
| DUNE ND-LAr / edep-sim chain | [`RobMina/bdnmc-dune`](https://github.com/RobMina/bdnmc-dune) |
| Mu2e Geant4-free fastsim | [`RobMina/mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim) |
| Fibre / WLS tables | [`RobMina/scint-fibre-fastsim`](https://github.com/RobMina/scint-fibre-fastsim) |
| Scoring / benches | [`RobMina/hep-simbench`](https://github.com/RobMina/hep-simbench) |

Do not merge generators into this repository. The detector simulation stays stock NOvASoft.
