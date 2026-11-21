# Site smoke (generate → HEPEVT input)

The detector simulation is stock NOvASoft (sim stage). This smoke documents the
**generation** stage that writes the HEPEVT input via `novafastsim_ldm submit`.

Measured job IDs, absolute site paths and counts are not published in this tree.

## Card requirement

Private detector frames set cuboid `card_overrides` (`width`, `det-theta`, …). Do **not**
embed a MiniBooNE sphere test card — `bdnmc_nova.generate` will `KeyError` after the
job starts. Use the public cuboid reference card in this tree:

`examples/data/generic_cuboid_pi0_m005.dat`

This card checks the plumbing only: generic 120 GeV beam on carbon, analytic BMPT π⁰,
unrotated cuboid at 500 m, placeholder normalisation. Its yields and kinematics are not
NOvA physics; use the campaign cards (g4numi `particle_list` production) for that.

## Site settings

Fill `site.env` from [`site.env.example`](../site.env.example). Required names
(`BDNMC_NOVA_SRC`, `GEN_IMAGE`, `HEPEP_*`, `BDNMC_SHARED`, …) are site-local; never
commit real values.

```bash
# from a nova-fastsim checkout
pip install -e core -e 'ldm[chains]'
cp site.env.example site.env   # edit: BDNMC_NOVA_SRC, GEN_IMAGE, HEPEP_*, …

# Private ND frame (never commit). Copy it from the site (location: on-site README)
# to a local file and point BDNMC_NOVA_CONFIG or --config at it, e.g.
# /tmp/nova_frame_private.json.
```

## Generate (CPU smoke; gpus=0)

GPU partitions often queue longer; CPU is enough for a small trial count.

```bash
python -m novafastsim_ldm submit --stages gen --handle nova-fs-smoke-gen \
  --card examples/data/generic_cuboid_pi0_m005.dat \
  --config /tmp/nova_frame_private.json \
  --endpoint slurm_ssh --gpus 0 --cores 4 --mem-gb 16 --minutes 30 \
  --trials 1048576 --precision float32 \
  --preamble 'module load apptainer'
```

Outputs land under `$HEPEP_SLURM_WORKDIR/<handle>/out/` (`events.txt`, `summary.json`,
`slurm.out`).

## Dry-run / site check

`check-site` validates what `submit --stages …` needs (embedded card / frame checked on this
machine; site paths such as `GEN_IMAGE` only required to be set). For a local run use
`generate --dry-run`.

```bash
python -m novafastsim_ldm check-site --stages gen \
  --card examples/data/generic_cuboid_pi0_m005.dat \
  --config /tmp/nova_frame_private.json

python -m novafastsim_ldm submit --stages gen --handle dry \
  --card examples/data/generic_cuboid_pi0_m005.dat \
  --config /tmp/nova_frame_private.json --dry-run
```

## Optional: GPU generate

```bash
# set HEPEP_SLURM_GPU_PARTITION (and related) in site.env
python -m novafastsim_ldm submit --stages gen --handle nova-fs-smoke-gpu \
  --card examples/data/generic_cuboid_pi0_m005.dat \
  --config /tmp/nova_frame_private.json \
  --endpoint slurm_ssh --gpus 1 \
  --trials 1048576 --precision float32 \
  --preamble 'module load apptainer'
```

## Sim stage (stock detector simulation)

Needs `NOVA_SETUP` and the simulation job configuration (`NOVA_FCL`; private inputs
bundle or a site-local path). `NOVA_IMAGE` is optional (unset: the simulation runs in the
job's own environment). Run it together with generation (`--stages gen,sim`), or alone on
an existing HEPEVT file:

```bash
python -m novafastsim_ldm submit --stages sim --handle nova-fs-smoke-sim \
  --events-at "$HEPEP_SLURM_WORKDIR/nova-fs-smoke-gen/out/events.txt" \
  --endpoint slurm_ssh --preamble 'module load apptainer'
# or --events events.txt [--max-events N] to embed a local file
```

See [`two_stage.md`](two_stage.md).
