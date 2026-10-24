"""Tests for src/pricing/heston.py: Fourier pricing vs Black-Scholes limit, parity and Monte Carlo."""

import pytest

from pricing.black_scholes import black_scholes_price
from pricing.heston import HestonParams, heston_mc_price, heston_price

S0, K, T, R, Q = 100.0, 100.0, 1.0, 0.03, 0.01
PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def test_reduces_to_black_scholes_when_vol_of_vol_is_tiny() -> None:
    """With xi -> 0 and v0 = theta, variance is constant and Heston collapses to BS."""
    flat = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-3, rho=0.0)
    bs = black_scholes_price(S0, K, T, R, 0.2, "call", Q)
    assert heston_price(S0, K, T, R, flat, "call", Q) == pytest.approx(bs, abs=1e-3)


def test_put_call_parity() -> None:
    import numpy as np

    c = heston_price(S0, K, T, R, PARAMS, "call", Q)
    p = heston_price(S0, K, T, R, PARAMS, "put", Q)
    assert c - p == pytest.approx(S0 * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-8)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_fourier_price_matches_monte_carlo(strike: float) -> None:
    exact = heston_price(S0, strike, T, R, PARAMS, "call", Q)
    mc = heston_mc_price(S0, strike, T, R, PARAMS, "call", Q, n_paths=200_000, n_steps=200)
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.03


def test_negative_rho_produces_downward_skew() -> None:
    """Rho < 0 makes low strikes more expensive in vol terms than high strikes."""
    from pricing.implied_vol import implied_volatility

    lo = implied_volatility(heston_price(S0, 80.0, T, R, PARAMS, "call", Q), S0, 80.0, T, R, "call", Q)
    hi = implied_volatility(heston_price(S0, 120.0, T, R, PARAMS, "call", Q), S0, 120.0, T, R, "call", Q)
    assert lo > hi


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=1.0)


def test_feller_flag() -> None:
    assert not HestonParams(0.04, 1.0, 0.04, 1.0, -0.5).feller_satisfied
    assert HestonParams(0.04, 3.0, 0.04, 0.3, -0.5).feller_satisfied
