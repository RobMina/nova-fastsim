# novafastsim_ldm

Thin orchestration over the existing bdnmc-nova chain (phase 2):

1. **generate** — BdNMC (JAX) → HEPEVT text file (`bdnmc_nova.generate`)
2. **ldmgen** (optional) — [`RobMina/ldmgen`](https://github.com/RobMina/ldmgen) `ldmgen/1` HDF5;
   `export nova` still uses the bdnmc-nova HEPEVT writer
3. **events-*** — validate / head the HEPEVT *input* text before sim submit
4. **simulate** — stock detector simulation via `bdnmc_nova.jobspec` (simulation image)

This package does **not** rewrite the detector simulation or merge generators into the tree.

```bash
pip install -e ../core -e '.[chains,ldmgen]'
python -m novafastsim_ldm describe
python -m novafastsim_ldm generate --trials 1024 --out events.txt
python -m novafastsim_ldm events-info events.txt
python -m novafastsim_ldm events-head events.txt --n 10 --out head.txt
python -m novafastsim_ldm submit --script-out /tmp/job.sh   # dry-run script
python -m novafastsim_ldm ldmgen-status
python -m novafastsim_ldm ldmgen-info rec.h5          # library API
python -m novafastsim_ldm ldmgen-export rec.h5 --out events.txt --config frame.json
```

The bridge calls `ldmgen.adapters.bdnmc` / `ldmgen.export.to_nova` directly (not CLI argv).


Site paths come from `site.env` (see repo `site.env.example`). Details:
[`../examples/README.md`](../examples/README.md).
