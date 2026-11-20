# UVA two-stage LDM chain (pointer)

Maintainer notes for the shared UVA area live in the private ai-config wiki
(`nova/wiki/bdnmc/uva-shared-area.md`). Collaborators use
`/project/hep_aag/bdnmc/README.md` on site.

Public packages involved:

1. [`RobMina/BdNMC`](https://github.com/RobMina/BdNMC) (`gpu` / bdnmc-jax)
2. [`RobMina/bdnmc-nova`](https://github.com/RobMina/bdnmc-nova) (frame, TextFileGen file, jobspec)
3. [`RobMina/hep-endpoints`](https://github.com/RobMina/hep-endpoints) (Slurm / EAF / grid)
4. Stock NOvASoft TextFileGen in the SL7 image

`nova-fastsim` will eventually own profile / site / examples that call into (2)–(3)
without merging generators. Until then, run the bdnmc-nova job directly.
