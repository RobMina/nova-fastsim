"""Synthetic tests: profile install and NOVA_DEVICE_FLOAT fail-closed behaviour."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_profile_installs_nova():
    from novafastsim import profile

    p = profile.install()
    assert p.name == "nova"
    assert profile.current().name == "nova"
    assert "nd" in p.subsystems


def test_dtype_compute_override():
    from novafastsim import dtype

    assert dtype.compute_dtype("float32") == np.dtype("float32")
    assert dtype.compute_dtype("float64") == np.dtype("float64")
    with pytest.raises(ValueError, match="float16"):
        dtype.compute_dtype("float16")


def test_dtype_rejects_ood_in_subprocess():
    env = {**os.environ, "NOVA_DEVICE_FLOAT": "float16", "PYTHONPATH": str(ROOT / "core")}
    # Fresh interpreter so dtype.py re-reads the environment at import.
    code = "import novafastsim.dtype"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "NOVA_DEVICE_FLOAT" in (proc.stderr + proc.stdout) or "float16" in (proc.stderr + proc.stdout)


def test_packages_import():
    import novafastsim_calib
    import novafastsim_ldm
    import novafastsim_beamline

    assert novafastsim_calib.__version__
    assert novafastsim_ldm.__version__
    assert novafastsim_beamline.__version__
