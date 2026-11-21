# Examples — LDM chain via `novafastsim_ldm`

Phase 2 wires a thin CLI over the proven bdnmc-nova path. Still no new
detector simulator in this repository.

```text
BdNMC (JAX, bdnmc-jax image)
  → python -m novafastsim_ldm generate     # → bdnmc_nova.generate → events.txt
  → [optional] ldmgen convert / export     # RobMina/ldmgen; pip install 'novafastsim-ldm[ldmgen]'
  → python -m novafastsim_ldm events-info  # validate the HEPEVT *input*
  → python -m novafastsim_ldm submit       # → bdnmc_nova.jobspec → stock detector simulation
```

## Orchestrate, do not rewrite

- The detector simulation that reads the HEPEVT file is stock NOvASoft. Never reimplemented here.
- Generators stay in BdNMC / bdnmc-nova / [`ldmgen`](https://github.com/RobMina/ldmgen).
- Private ND frame / job configuration: site-local paths today (`$BDNMC_SHARED`); later
  a private inputs bundle ([`setup/README.md`](../setup/README.md)).
- Two images: JAX host + NOvASoft SL7.

## Site.env

Copy [`site.env.example`](../site.env.example) → `site.env` and fill paths as the
job sees them. Required names fail closed via `novafastsim.site.require`.

## Commands

```bash
pip install -e core -e 'ldm[chains,ldmgen]'
python -m novafastsim_ldm describe
python -m novafastsim_ldm ldmgen-status

# generation only (needs bdnmc-jax + private frame JSON)
python -m novafastsim_ldm generate --trials 65536 --out events.txt --summary summary.json

# validate HEPEVT before sim-only submit
python -m novafastsim_ldm events-info events.txt

# dry-run: write the embedded two-stage job script (mode 0700; contains private frame)
python -m novafastsim_ldm submit --script-out /tmp/nova-fastsim-job.sh

# real submit (hep-endpoints site settings: HEPEP_*)
python -m novafastsim_ldm submit --gpus 1

# calib scaffolding (parallel track)
python -m novafastsim_calib describe
python -m novafastsim_calib status
```

Direct bdnmc-nova entry points remain valid; this CLI only fills site defaults:

| Piece | Where |
|---|---|
| Frame + HEPEVT writer | `bdnmc_nova.frame` and the bdnmc-nova HEPEVT writer |
| Stage-1 generate | `python -m bdnmc_nova.generate …` |
| Job wrapper | `jobs/two_stage.sh` / `python -m bdnmc_nova.jobspec …` |
| Collaborator shared root | `$BDNMC_SHARED` (site-local) |

Also see [`uva_two_stage.md`](uva_two_stage.md).
