# Examples — use the existing bdnmc-nova / UVA chain

v0 does not ship a new detector simulator. The proven path is:

```text
BdNMC (JAX, bdnmc-jax image)
  → bdnmc_nova.generate / textfilegen   # writes HEPEVT for TextFileGen
  → NOvASoft TextFileGen (stock art)    # SL7 image; MCTruth → G4 → reco
```

That two-stage job is already packaged in
[`RobMina/bdnmc-nova`](https://github.com/RobMina/bdnmc-nova):

| Piece | Where |
|---|---|
| Frame + HEPEVT writer | `bdnmc_nova.frame`, `bdnmc_nova.textfilegen` |
| Stage-1 generate | `python -m bdnmc_nova.generate …` |
| One-job wrapper | `jobs/two_stage.sh` (`bdnmc_nova/data/two_stage.sh`) |
| hep-endpoints submit | `python -m bdnmc_nova.jobspec submit …` |
| Collaborator area (UVA) | `$BDNMC_SHARED` (see private workspace wiki `<private-nova-wiki>/bdnmc/uva-shared-area`) |

## Orchestrate, do not rewrite

- **TextFileGen** = stock NOvASoft. Do not reimplement it in JAX or in this repo.
- Private ND frame / FCLs stay out of the public tree (future `nova-fastsim-inputs`).
- Two images: JAX host (`container/nova-fastsim-jax.def` here, or `bdnmc-jax`) plus NOvASoft SL7.

## Minimal pointer (after site.env)

Fill `site.env` from `site.env.example`, then on a host with Apptainer + the shared area:

```bash
# Generation + TextFileGen sim as one Slurm job (bdnmc-nova API — illustrative)
python -m bdnmc_nova.jobspec submit "$CARD" \
  --handle nova-fastsim-example \
  --endpoint slurm_ssh \
  --config "$BDNMC_NOVA_CONFIG" \
  --src "$BDNMC_NOVA_SRC" \
  --gen-image "$GEN_IMAGE" \
  --nova-image "$NOVA_IMAGE" \
  --nova-setup "$NOVA_SETUP" \
  --nova-fcl "$NOVA_FCL" \
  --gpus 1
```

Or the shell form from bdnmc-nova:

```bash
GEN_IMAGE=… NOVA_IMAGE=… NOVA_SETUP=… NOVA_FCL=… CARD=… \
  BDNMC_NOVA_CONFIG=… GPU=1 bash "$BDNMC_NOVA_SRC/jobs/two_stage.sh"
```

Thin `novafastsim_ldm` wrappers that read `site.env` and call the same entry points
are planned for a later milestone; this example stays a documented pointer.
