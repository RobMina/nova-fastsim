# novafastsim_calib

Scaffolding for NOvA ND calibration via hep-simlib (recorder → banks under the
`nova` profile → `hepsimlib.fit` / expected). Runs **in parallel** with the LDM
example track. Fibre / WLS tables come from
[`scint-fibre-fastsim`](https://github.com/RobMina/scint-fibre-fastsim) when the
optional extra is installed; lumen cell optics stay parked.

```bash
pip install -e ../core -e .
python -m novafastsim_calib describe
python -m novafastsim_calib status
python -m novafastsim_calib ensure-dirs   # creates empty subsystem dirs under NOVA_LIBRARY_CACHE
```

Bank layout is env-driven (`NOVA_LIBRARY_CACHE`, `NOVA_DEVICE_FLOAT`). No
site-absolute paths in this package. Generators and stock TextFileGen stay in
`novafastsim_ldm` / NOvASoft — not here.
