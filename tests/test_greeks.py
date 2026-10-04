"""Tests validating src/pricing/greeks.py: analytical Greeks vs finite differences."""

import pytest

from pricing.black_scholes import black_scholes_price
from pricing.greeks import analytical_greeks, finite_difference_derivative

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type: str) -> None:
    """Analytical delta must match a central finite difference on S."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    base_kwargs = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type, q=Q)
    fd_delta = finite_difference_derivative(
        black_scholes_price, base_kwargs, param_name="S", h=0.01, order=1
    )
    assert greeks.delta == pytest.approx(fd_delta, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(option_type: str) -> None:
    """Analytical gamma must match a central second finite difference on S.

    Gamma needs a coarser step than delta: the second difference divides by
    h**2 and amplifies floating-point rounding error, so a very small h would
    make the finite-difference estimate noisier, not more accurate.
    """
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    base_kwargs = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type, q=Q)
    fd_gamma = finite_difference_derivative(
        black_scholes_price, base_kwargs, param_name="S", h=0.5, order=2
    )
    assert greeks.gamma == pytest.approx(fd_gamma, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type: str) -> None:
    """Analytical vega must match a central finite difference on sigma."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    base_kwargs = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type, q=Q)
    fd_vega = finite_difference_derivative(
        black_scholes_price, base_kwargs, param_name="sigma", h=1e-4, order=1
    )
    assert greeks.vega == pytest.approx(fd_vega, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type: str) -> None:
    """Analytical rho must match a central finite difference on r."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    base_kwargs = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type, q=Q)
    fd_rho = finite_difference_derivative(
        black_scholes_price, base_kwargs, param_name="r", h=1e-4, order=1
    )
    assert greeks.rho == pytest.approx(fd_rho, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type: str) -> None:
    """Analytical theta (-dPrice/dT) must match minus a finite difference on T."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    base_kwargs = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type=option_type, q=Q)
    fd_dprice_dT = finite_difference_derivative(
        black_scholes_price, base_kwargs, param_name="T", h=1e-4, order=1
    )
    assert greeks.theta == pytest.approx(-fd_dprice_dT, abs=1e-3)


def test_call_delta_is_between_zero_and_one() -> None:
    """A call's delta must lie in (0, 1) for a standard ATM-ish setup."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type="call", q=Q)
    assert 0.0 < greeks.delta < 1.0


def test_put_delta_is_between_minus_one_and_zero() -> None:
    """A put's delta must lie in (-1, 0) for a standard ATM-ish setup."""
    greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type="put", q=Q)
    assert -1.0 < greeks.delta < 0.0


def test_gamma_is_positive_and_equal_for_call_and_put() -> None:
    """Gamma must be positive, and identical for a call and a put (same strike/maturity)."""
    call_greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type="call", q=Q)
    put_greeks = analytical_greeks(S0, K, T, R, SIGMA, option_type="put", q=Q)
    assert call_greeks.gamma > 0.0
    assert call_greeks.gamma == pytest.approx(put_greeks.gamma)


@pytest.mark.parametrize(
    "S, K_, T_, sigma",
    [
        (-10.0, K, T, SIGMA),
        (S0, 0.0, T, SIGMA),
        (S0, K, -1.0, SIGMA),
        (S0, K, T, 0.0),
    ],
)
def test_invalid_inputs_raise_value_error(S: float, K_: float, T_: float, sigma: float) -> None:
    """Non-positive S, K, T or sigma must raise ValueError."""
    with pytest.raises(ValueError):
        analytical_greeks(S, K_, T_, R, sigma)


def test_invalid_option_type_raises_value_error() -> None:
    """An option_type other than 'call'/'put' must raise ValueError."""
    with pytest.raises(ValueError):
        analytical_greeks(S0, K, T, R, SIGMA, option_type="straddle")
