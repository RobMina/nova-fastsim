# Two-stage LDM chain (pointer)

Site paths are not published here. Collaborators use the on-site README at
`$BDNMC_SHARED/README.md` (set `BDNMC_SHARED` to that site's collaborator root; never
hard-code it in public docs).

Public packages involved:

1. [`RobMina/BdNMC`](https://github.com/RobMina/BdNMC) (`gpu` / bdnmc-jax)
2. [`RobMina/bdnmc-nova`](https://github.com/RobMina/bdnmc-nova) (frame, HEPEVT file, jobspec)
3. [`RobMina/hep-endpoints`](https://github.com/RobMina/hep-endpoints) (Slurm / EAF / grid)
4. Stock NOvASoft detector simulation in its SL7 image

`nova-fastsim` owns profile / site / examples that call into (2)–(3) without merging
generators. Fill `site.env` and run `python -m novafastsim_ldm submit`.
