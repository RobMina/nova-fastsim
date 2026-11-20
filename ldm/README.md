# novafastsim_ldm

Thin orchestration over the existing UVA / bdnmc-nova chain:

1. BdNMC (JAX) generation → HEPEVT / TextFileGen file (`bdnmc_nova.generate` /
   `bdnmc_nova.textfilegen`, or `ldmgen export nova`)
2. Stock NOvASoft **TextFileGen** → detsim → reco → CAF (SL7 image)

This package will wrap site paths and jobspecs; it will **not** rewrite TextFileGen
or merge generators into this tree. See [`../examples/README.md`](../examples/README.md).
