# novafastsim_ldm

Thin orchestration over the existing UVA / bdnmc-nova chain (phase 2):

1. **generate** — BdNMC (JAX) → HEPEVT / TextFileGen file (`bdnmc_nova.generate`)
2. **ldmgen** (optional) — [`RobMina/ldmgen`](https://github.com/RobMina/ldmgen) `ldmgen/1` HDF5;
   `export nova` still calls `bdnmc_nova.textfilegen`
3. **simulate** — stock NOvASoft **TextFileGen** via `bdnmc_nova.jobspec` (SL7 image)

This package does **not** rewrite TextFileGen or merge generators into the tree.

```bash
pip install -e ../core -e '.[chains,ldmgen]'
python -m novafastsim_ldm describe
python -m novafastsim_ldm generate --trials 1024 --out events.txt
python -m novafastsim_ldm submit --script-out /tmp/job.sh   # dry-run script
python -m novafastsim_ldm ldmgen-status
```

Site paths come from `site.env` (see repo `site.env.example`). Details:
[`../examples/README.md`](../examples/README.md).
