"""Tests for src/pricing/sabr.py."""

import numpy as np
import pytest

from pricing.sabr import SabrParams, calibrate_sabr, sabr_implied_vol

F, T = 100.0, 1.0


def test_atm_vol_is_close_to_alpha_over_f_power() -> None:
    p = SabrParams(alpha=0.2 * F ** 0.5, beta=0.5, rho=-0.3, nu=0.4)
    atm = sabr_implied_vol(F, F, T, p)
    assert atm == pytest.approx(0.2, rel=0.03)


def test_nu_zero_limit_is_cev_flat_in_rho() -> None:
    """With negligible vol-of-vol, rho has no effect on the smile."""
    a = sabr_implied_vol(F, 90.0, T, SabrParams(2.0, 0.5, -0.5, 1e-6))
    b = sabr_implied_vol(F, 90.0, T, SabrParams(2.0, 0.5, 0.5, 1e-6))
    assert a == pytest.approx(b, rel=1e-4)


def test_calibration_recovers_known_parameters() -> None:
    true = SabrParams(alpha=2.0, beta=0.5, rho=-0.4, nu=0.6)
    strikes = np.linspace(70, 140, 15)
    vols = sabr_implied_vol(F, strikes, T, true)
    fit = calibrate_sabr(F, strikes, vols, T, beta=0.5)
    assert fit.alpha == pytest.approx(true.alpha, rel=1e-3)
    assert fit.rho == pytest.approx(true.rho, abs=1e-3)
    assert fit.nu == pytest.approx(true.nu, rel=1e-3)


def test_negative_rho_gives_downward_sloping_smile() -> None:
    p = SabrParams(2.0, 0.5, -0.6, 0.5)
    assert sabr_implied_vol(F, 80.0, T, p) > sabr_implied_vol(F, 120.0, T, p)


def test_calibration_needs_enough_points() -> None:
    with pytest.raises(ValueError):
        calibrate_sabr(F, np.array([90.0, 110.0]), np.array([0.2, 0.2]), T)
