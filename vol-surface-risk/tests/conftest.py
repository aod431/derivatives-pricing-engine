"""Shared fixtures: everything runs offline on the committed synthetic snapshot."""

from pathlib import Path

import pytest

from volrisk.data import load_snapshot, prepare_chain
from volrisk.surface import build_surface

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def chain():
    return load_snapshot(DATA / "sample_SPY.csv", source="sample")


@pytest.fixture(scope="session")
def prepared(chain):
    return prepare_chain(chain)


@pytest.fixture(scope="session")
def surface(prepared):
    return build_surface(prepared)
