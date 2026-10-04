"""Smoke test: the pricing package must be importable after installation."""

import pricing


def test_import_and_version() -> None:
    """The package should expose a __version__ string."""
    assert isinstance(pricing.__version__, str)
    assert pricing.__version__ != ""
