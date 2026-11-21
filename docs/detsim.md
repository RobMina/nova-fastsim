# On-device, differentiable detector simulation (`novafastsim_detsim`)

Goal: the whole Near Detector simulation — primary particles to read-out cells to event-level reconstruction —
runs on the accelerator in JAX, in float32 or float64, and is differentiable end to end: in the response
parameters (light yield, attenuation, quenching, electronics), in the shower and range model parameters, and in
the particle kinematics (vertex, direction, energy). The stock collaboration simulation is the validation
reference for this package, not part of its runtime.

## Who owns what

| Layer | Owner |
|---|---|
| Particles, deposits (mean tracks and showers), plane/cell geometry, energy sharing, reco, calibration fits | this package |
| Cell definition (`Cell`, `nova_cell_tdr`), cell optics (Birks quenching, looped-fibre transport, `CellTable`), NOvA APD + front end (expected, sampled, `PhotosensorBackend`) | [scint-fibre-fastsim](https://github.com/RobMina/scint-fibre-fastsim): `geometry.cell`, `response.cell`, `backends.nova_apd` |
| Fitting and gradient-check tooling | hep-simlib (`fit`, `testing`) |

Parameters `optics.<field>` and `apd.<field>` are scint-fibre-fastsim's `CellOptics` and `APDParams` fields in its
units (mm, ns, MeV); the detector config passes them through unchanged. The default cell cross-section, wall and
fibre come from `nova_cell_tdr()`.

## Chain

```text
Particles [N]            from_bdnmc (generator rows, on device) | from_hepevt (text) | make (arrays)
   │ deposits            mean track (range table) + mean EM shower (quadrature) → Segments [S]
   ▼
Segments [S]             p0, p1, edep, t0, t1, kind, mask
   │ geometry.pieces     segment ∩ (plane window × cell window) → path length, along-cell distance
   ▼
pieces [S, K]            mass-weighted energy share per cell
   │ scintfibrefastsim.response.light   Birks quenching, looped-fibre collection and delay (analytic or CellTable)
   │ × apd.qe, scatter-add
   ▼
Response [P, C]          edep, evis, expected PE, light-weighted time, expected DCS peak, hit probability
   │ nova_apd.expected   expected DCS peak, hit probability     (scintfibrefastsim.backends)
   │ nova_apd.sample     Poisson PE, APD excess noise, per-sample noise, DCS threshold, time jitter
   ▼
Hits [P, C]              pe, adc, time, hit
   │ reco                attenuation-corrected calorimetric energy, view regressions → direction, θ²
   ▼
event quantities         E, θ, Eθ²
```

Everything has fixed shapes, so one event is one `jit` call and a batch is `vmap` over a leading axis
(`expected_batch`, `sampled_batch`). `expected` is deterministic; `sampled` draws around it and its mean is the
expected response (tested). Fits use the expected response with a Poisson likelihood (or the hit probability as a
detection term) and never differentiate the sampled chain, following the hep-simlib calibration design.

## Models

| Stage | Model | Differentiable in |
|---|---|---|
| Geometry | planes of extruded cells, views alternating; scintillator interior `cell_width × cell_depth`, inert webs and faces; path length = overlap of three parameter intervals | segment endpoints (piecewise linear: kinks where an endpoint crosses a wall) |
| Tracks (μ, π, K, p) | straight line; mean CSDA loss from a Bethe + Sternheimer table in scintillator-equivalent column density (`rho_stop`); no multiple scattering, no hadronic interactions, no radiative loss | vertex, direction, energy, `rho_stop` |
| EM showers (e±, γ) | gamma longitudinal profile, t_max = ln(E/E_c) ∓ 0.5 (Longo–Sestili form); two-component radial profile of the Grindhammer–Peters form in Molière radii; fixed quadrature (`n_long × 2 n_rad × n_phi` short segments along the axis) | vertex, direction, energy, `x0`, `ec`, `rm`, `shower_b`, `z_eff` |
| Energy sharing | a segment's energy is shared over the materials it crosses in proportion to density × relative stopping power | `rho_*`, `stop_inert` |
| Quenching (scint-fibre-fastsim) | Birks–Chou with `optics.birks_kb`, `optics.birks_kc`; tracks use their own dE/dx, shower segments `shower_dedx` | all three |
| Fibre (scint-fibre-fastsim) | looped fibre, both ends on one pixel: half the light travels `d`, half `fibre_length − d`; two-exponential attenuation; light-weighted delay. Or a `CellTable` (`params["optics_table"]`): collection and delay vs distance, interpolated | analytic: `optics.*`, along-cell position; table: table values, `optics.yield_per_mev`, position |
| Channel efficiency | optional `channel_eff [P, C]` multiplies the light | each channel |
| APD + front end (scint-fibre-fastsim) | QE; unit-peak CR-RC shaper, samples every `tick`, dual-correlated sampling with `dcs_lag`; expected DCS peak; hit probability = average over the charge fluctuation (gamma-shaped, variance F·PE; Gauss–Hermite through Wilson–Hilferty) of P(any DCS sample > threshold) with per-sample noise. Agrees with the sampled hit rate within 0.03 near threshold | `apd.*`, hit time |
| Reco | per view weighted regression of u on z → direction; each view's cells attenuation-corrected at the other view's centroid → visible energy; `reco_scale` | the charges, hence everything above |

## Detector description

`Detector(geometry, settings)` is static (a jit argument); parameters are a dict of arrays. Defaults are a
public caricature: cell interior 3.9 × 6.6 cm, 0.7 mm looped fibre and the ND front-end sampling of public papers
(e.g. arXiv:1807.02665); generic liquid scintillator, PVC, fibre and photodetector constants. They are **not** the
collaboration's geometry, materials or calibration. The private inputs bundle provides a detector-config JSON:

```json
{"geometry": {"n_planes": 192, "n_cells": 96, "cell_width": 3.9, "readout_end": [1, 1]},
 "settings": {"n_long": 64},
 "params": {"optics.yield_per_mev": 11.8, "optics.atten_long": 7000.0, "apd.threshold_pe": 10.0}}
```

`config.load(path)` fails closed on unknown keys. `python -m novafastsim_detsim describe [--config F]` prints the
resolved description.

## Validation status

| Check | State |
|---|---|
| Geometry: axial and transverse exact cases; random-segment scintillator fraction; window overflow | pytest |
| Energy closure: contained track = kinetic energy; contained shower = kinetic energy; scintillator + inert = total | pytest |
| Shower depth: mean depth = (a/b) X0; photon − electron = 1 X0 | pytest |
| Sampled mean = expected (PE); sampled hit rate = Σ hit probability | pytest |
| float32 vs float64 | pytest (PE to 1e-3) |
| Autodiff vs central differences: kinematics, response parameters, shower parameters | pytest (hepsimlib.testing) |
| Calibration closure: light yield and long attenuation from Poisson-sampled muons (hepsimlib.fit) | pytest |
| Cell response = scint-fibre-fastsim `light` × QE cell by cell; `CellTable` path = analytic path | pytest |
| Cell optics and APD (FD gradients, sampled vs expected, backend over arrivals) | scint-fibre-fastsim pytest |
| Cell optics and front-end timing against public NOvA data (Test Beam cell, fibre attenuation curve, a Far Detector cell, ND and FD timing resolution), criteria declared first | scint-fibre-fastsim [`validation/nova_cell`](https://github.com/RobMina/scint-fibre-fastsim/tree/main/validation/nova_cell) and the NOVA-CELL benchmark. Train references pass. Test references fail: the fibre-spool attenuation over-attenuates a 15 m cell, and timing between 30 and 400 PE is too optimistic (see limitations 4 and 8) |
| Reco: attenuation correction closes on visible energy; θ² within 25 % for a 0.15 rad shower | pytest |
| Against the stock simulation (per-cell PE, hit counts, shower profiles, calorimetric energy, θ²) | **not done**: needs a hep-simbench workflow on stock ND samples |
| Accelerator throughput | laptop CPU: ~160 events/s (sampled chain + reco, 18 432 channels, both dtypes); GPU not measured |

## Limitations (in order of what to replace)

1. **Mean showers only.** No shower-to-shower fluctuations; the expected profile is right on average but the
   sampled readout only fluctuates in the photodetector. Next: a stochastic tier from Geant4-matched kernels
   (hep-simkernels e± stepping, plus photon processes still to add there) or a recorded hep-simlib component
   bank, sharing the `Segments` interface. The mean model stays the differentiable surrogate.
2. **Tracks are straight, mean-loss, non-interacting.** Hadronic interactions of π and p, multiple scattering and
   δ-rays are absent; muon radiative losses are ignored.
3. **Single hit per cell.** Pulses from separate times in one cell are merged at the light-weighted time; no
   pile-up, no noise hits from cells without signal in the expected form, no dead channels beyond `channel_eff`.
4. **Cell optics are analytic, and their defaults are not tuned to NOvA.**
   - The `CellTable` path is in place. The table should come from scint-fibre-fastsim's photon Monte Carlo of the
     NOvA cell (liquid scintillator, PVC walls, looped Y11 fibre). That needs the kernel to model a cell; its
     transport is built for the extruded strip today.
   - Against public data, the looped two-exponential form fits a 2.6 m and a 15 m cell, but with an in-cell
     attenuation (about 40 % short component, 2 m and 10 m). The fibre-spool curve over-attenuates a 15 m cell.
   - The package defaults give 0.37 of the Test Beam light, so a detector config must set `optics.*`.
5. **Hard cell walls** make the response piecewise linear in segment positions. Gradients are exact almost
   everywhere but rough at the cell scale; an optional wall-smoothing width is a planned knob for optimisation
   problems that need smooth objectives.
6. **No muon catcher**, no inactive regions, no survey alignment. Per-plane offsets are the next differentiable
   geometry parameters.
7. **Reco is a stand-in.** It gives differentiable E and θ² for sensitivity and design studies, not the
   collaboration's reconstruction.
8. **Sampled hit times are too precise at low charge.**
   - The sampler jitters times by `apd.time_res / sqrt(PE)`. The default 10 ns gives 2.6 ns at 15 PE, where the ND
     measures about 27 ns.
   - scint-fibre-fastsim's `nova_apd.timing_crb` (the front-end Cramér–Rao bound) reproduces the measured
     resolution at 12.5 PE in both detectors, but is about a factor 2 low between 60 and 200 PE.
   - A timing model that covers that range, and wiring it into the sampler, are open.

## References (public)

- E. Longo and I. Sestili, Nucl. Instrum. Meth. 128 (1975) 283 — longitudinal profile.
- G. Grindhammer and S. Peters, arXiv:hep-ex/0001020 — parametrised showers.
- Particle Data Group, *Passage of particles through matter* — Bethe formula, Sternheimer density effect,
  material constants.
- NOvA public papers on detector construction and electronics (e.g. arXiv:1807.02665) — cell and readout scales.
