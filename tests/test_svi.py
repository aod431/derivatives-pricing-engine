"""Tests for src/pricing/svi.py."""

import numpy as np
import pytest

from pricing.svi import (
    SviParams,
    butterfly_g,
    calendar_violations,
    fit_svi,
    has_butterfly_arbitrage,
    svi_total_variance,
)

GOOD = SviParams(a=0.02, b=0.1, rho=-0.4, m=0.0, s=0.2)


def test_minimum_variance_location() -> None:
    k = np.linspace(-1, 1, 2001)
    w = svi_total_variance(k, GOOD)
    # With rho<0 the minimum sits at k > m.
    assert k[np.argmin(w)] > GOOD.m


def test_good_params_have_no_butterfly_arbitrage() -> None:
    assert not has_butterfly_arbitrage(GOOD)


def test_steep_wings_create_butterfly_arbitrage() -> None:
    """Wing slope b(1+rho) = 5.7 is far above Lee's bound of 2 and breaks g(k)."""
    bad = SviParams(a=0.0, b=3.0, rho=0.9, m=0.0, s=0.01)
    assert has_butterfly_arbitrage(bad)


def test_calendar_violation_detected() -> None:
    low = SviParams(0.05, 0.1, -0.3, 0.0, 0.2)
    high = SviParams(0.02, 0.1, -0.3, 0.0, 0.2)
    assert calendar_violations([(0.5, low), (1.0, high)]) == [(0.5, 1.0)]
    assert calendar_violations([(0.5, high), (1.0, low)]) == []


def test_fit_recovers_synthetic_slice() -> None:
    k = np.linspace(-0.5, 0.5, 21)
    w = svi_total_variance(k, GOOD)
    fit = fit_svi(k, w)
    assert np.max(np.abs(svi_total_variance(k, fit) - w)) < 1e-4
    assert not has_butterfly_arbitrage(fit)


def test_fit_validates_input() -> None:
    with pytest.raises(ValueError):
        fit_svi(np.array([0.0, 0.1]), np.array([0.04, 0.05]))


def test_g_is_finite_everywhere() -> None:
    assert np.all(np.isfinite(butterfly_g(np.linspace(-2, 2, 50), GOOD)))
