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


def test_dtype_resolve_default(monkeypatch):
    from novafastsim import dtype

    monkeypatch.delenv("NOVA_DEVICE_FLOAT", raising=False)
    assert dtype.resolve() == "float64"
    monkeypatch.setenv("NOVA_DEVICE_FLOAT", "float32")
    assert dtype.resolve() == "float32"


def test_dtype_lazy_float_attr(monkeypatch):
    from novafastsim import dtype

    monkeypatch.setenv("NOVA_DEVICE_FLOAT", "float32")
    assert dtype.FLOAT == np.dtype("float32")
    assert dtype.SINGLE is True


def test_dtype_rejects_ood_via_resolve(monkeypatch):
    from novafastsim import dtype

    monkeypatch.setenv("NOVA_DEVICE_FLOAT", "float16")
    with pytest.raises(ValueError, match="NOVA_DEVICE_FLOAT"):
        dtype.resolve()


def test_inputs_module_survives_bad_dtype_env(monkeypatch):
    """Importing novafastsim.inputs must not fail on a bad NOVA_DEVICE_FLOAT."""
    env = {**os.environ, "NOVA_DEVICE_FLOAT": "float16", "PYTHONPATH": str(ROOT / "core")}
    code = "import novafastsim.inputs; print('ok')"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_packages_import():
    import novafastsim_calib
    import novafastsim_ldm
    import novafastsim_beamline

    assert novafastsim_calib.__version__
    assert novafastsim_ldm.__version__
    assert novafastsim_beamline.__version__
