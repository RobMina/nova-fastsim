# Private inputs release

Collaboration-internal geometry, job configuration, frame JSON and conditions are
**not** in the public `nova-fastsim` tree. They will ship as a separate private
bundle that this package verifies against a manifest.

## Status

| Item | State |
|---|---|
| Public code (`RobMina/nova-fastsim`) | live (v0+) |
| Private bundle | location and tag TBD |
| First release tag | not cut yet |
| Env var | `NOVA_FASTSIM_INPUTS` (root of a verified release) |

Until a release exists, LDM jobs read their private configuration from site paths
set in `site.env` (`BDNMC_NOVA_CONFIG`, `NOVA_FCL`, `$BDNMC_SHARED`). Those paths
are site-local and are not published here.

## Planned first-bundle contents

The exact file list is still open. Expected categories:

| Category | Examples | Redistribution |
|---|---|---|
| Detector frame | bdnmc-nova frame JSON (position / orientation) | collaboration-internal |
| Simulation job configuration | detector simulation and reconstruction configuration | collaboration-internal |
| Geometry / materials | surveyed snippets needed by fast response | collaboration-internal |
| Optional tables | fibre / photosensor response when calib lands | collaboration-internal |

Public meson lists and cards stay in public generators (`g4numi-ldm-tools`, BdNMC
cards without private offsets). Do not put analysis results in the inputs bundle.

## Manifest shape

[`RELEASE.example.json`](RELEASE.example.json) shows schema version 1: relative
`path`, `sha256`, `size_bytes`, `provenance`, `redistribution` (`nova-internal` or
`approved`). Real manifests must not be committed here.

## Verify

Schema check (no private files; works on the example):

```bash
python -m novafastsim.inputs validate setup/RELEASE.example.json
```

Against a release root (placeholders in the example report `missing`):

```bash
export NOVA_FASTSIM_INPUTS=/path/to/inputs-release
python -m novafastsim.inputs check /path/to/manifest.json --root "$NOVA_FASTSIM_INPUTS"
```

## Requesting access

Collaborators: ask the maintainers for the release root and manifest path. Paths are
not published in this repository.
