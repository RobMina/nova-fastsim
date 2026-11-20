# benchmarks/

Phase-2 stub for [hep-simbench](https://github.com/RobMina/hep-simbench) on the NOvA ND track.

```bash
python benchmarks/hepsimbench_nd.py status
# when hep-simbench is installed:
python benchmarks/hepsimbench_nd.py synthetic --out /tmp/nova-hepsimbench-example
```

Real ND histogram converters and workflow tables land with calib / on-device response
(phases 3–4). Scoring stays in hep-simbench; this repo only supplies ND layout helpers.
