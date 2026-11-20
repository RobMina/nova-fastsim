# Private inputs release (`nova-fastsim-inputs`)

NOvA-internal geometry, FCLs, frame JSON and conditions are **not** in the public
`nova-fastsim` tree. They will ship as a separate private bundle, parallel to
`mu2e-fastsim-inputs`.

## Status

| Item | State |
|---|---|
| Public code (`RobMina/nova-fastsim`) | live (v0+) |
| Private bundle name | **`nova-fastsim-inputs`** (repo / tag TBD) |
| First release tag | not cut yet |
| Env var | `NOVA_FASTSIM_INPUTS` (root of a verified release) |

Until a release exists, LDM jobs keep using the UVA shared-area private paths
(`BDNMC_NOVA_CONFIG`, simulation FCLs under `/project/hep_aag/bdnmc/nova/private/`
or the equivalent FNAL paths). See ai-config `nova/wiki/bdnmc/uva-shared-area.md`.

## Planned first-bundle contents

Exact file list is still open (plan open question). Expected categories:

| Category | Examples | Redistribution |
|---|---|---|
| ND frame | bdnmc-nova frame JSON (position / orientation) | NOvA-internal |
| Simulation FCLs | TextFileGen → detsim → reco → CAF | NOvA-internal |
| Geometry / materials | surveyed ND snippets needed by fast response | NOvA-internal |
| Optional tables | fibre / APD response when calib lands | NOvA-internal |

Public meson lists and cards stay in public generators (`g4numi-ldm-tools`, BdNMC
cards without private offsets). Do not put analysis results in the inputs bundle.

## Manifest shape

[`RELEASE.example.json`](RELEASE.example.json) shows schema version 1 (same idea as
mu2e-fastsim `setup/RELEASE.example.json`): relative `path`, `sha256`, `size_bytes`,
`provenance`, `redistribution: nova-internal`. Real manifests must not be committed
here.

## Verify (when a release exists)

```bash
export NOVA_FASTSIM_INPUTS=/path/to/nova-fastsim-inputs-vN
# checker lands with the first release (novafastsim.inputs); until then verify
# sha256sums from the private manifest by hand.
```

## Requesting access

Fermilab NOvA / UVA `hep_aag` collaborators: ask the maintainers for the release
root and manifest path. Paths are not published in this repository.
