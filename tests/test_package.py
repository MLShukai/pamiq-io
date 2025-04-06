"""Tests package top level features."""

import tomllib
from pathlib import Path

import pamiq_io

PROJECT_ROOT = Path(__file__).parent.parent


def test_version():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert pamiq_io.__version__ == pyproject["project"]["version"]
