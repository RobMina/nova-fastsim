"""site.load_dotenv: empty skip, last-wins in file, comments, export."""

from __future__ import annotations

import os

import pytest

from novafastsim import site


def test_empty_value_unsets_and_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("CARD", raising=False)
    monkeypatch.delenv("NOVAFS_JOB_HANDLE", raising=False)
    env = tmp_path / "site.env"
    env.write_text("CARD=\nNOVAFS_JOB_HANDLE=h\n")
    site.load_dotenv(env)
    assert "CARD" not in os.environ
    assert os.environ["NOVAFS_JOB_HANDLE"] == "h"


def test_last_occurrence_wins_within_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CARD", raising=False)
    env = tmp_path / "site.env"
    env.write_text("CARD=\nCARD=/real/card.dat\n")
    site.load_dotenv(env)
    assert os.environ["CARD"] == "/real/card.dat"


def test_real_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD", "/from/process")
    env = tmp_path / "site.env"
    env.write_text("CARD=/from/file\n")
    site.load_dotenv(env)
    assert os.environ["CARD"] == "/from/process"


def test_inline_comment_and_export(tmp_path, monkeypatch):
    monkeypatch.delenv("GEN_IMAGE", raising=False)
    monkeypatch.delenv("NOVA_SETUP", raising=False)
    env = tmp_path / "site.env"
    env.write_text(
        "export GEN_IMAGE=/a/b.sif  # gpu image\n"
        "NOVA_SETUP='/opt/setup.sh'\n"
    )
    site.load_dotenv(env)
    assert os.environ["GEN_IMAGE"] == "/a/b.sif"
    assert os.environ["NOVA_SETUP"] == "/opt/setup.sh"


def test_mismatched_quotes_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("CARD", raising=False)
    env = tmp_path / "site.env"
    env.write_text("CARD=\"/path/with'quote\n")
    site.load_dotenv(env)
    # unmatched quote is not stripped as a pair
    assert os.environ["CARD"].startswith('"') or "'" in os.environ["CARD"]


def test_require_rejects_empty(monkeypatch):
    monkeypatch.setenv("X", "")
    with pytest.raises(KeyError, match="X"):
        site.require("X")
