"""Tests validating src/pricing/monte_carlo.py against Black-Scholes and its own statistics."""

import pytest

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import mc_price

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02


@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_price_within_three_standard_errors_of_black_scholes(antithetic: bool) -> None:
    """The MC estimate must fall within 3 standard errors of the closed-form BS price.

    This is a statistical test, not an exact-equality one: with a large enough
    N, a well-implemented estimator should land inside a 3-SE band around the
    true value in the overwhelming majority of runs (fixed seed makes it
    deterministic here).
    """
    bs_price = black_scholes_price(S0, K, T, R, SIGMA, option_type="call", q=Q)
    result = mc_price(
        S0, K, T, R, SIGMA, option_type="call", q=Q,
        n_paths=200_000, antithetic=antithetic, seed=42,
    )
    assert abs(result.price - bs_price) < 3 * result.std_error


def test_confidence_interval_bounds_are_consistent() -> None:
    """ci_lower < price < ci_upper, and the CI half-width matches 1.96 * std_error."""
    result = mc_price(S0, K, T, R, SIGMA, option_type="call", q=Q, n_paths=50_000, seed=42)
    assert result.ci_lower < result.price < result.ci_upper
    assert result.ci_upper - result.price == pytest.approx(1.96 * result.std_error)
    assert result.price - result.ci_lower == pytest.approx(1.96 * result.std_error)


@pytest.mark.parametrize(
    "S, K_, T_, sigma, n_paths",
    [
        (-10.0, K, T, SIGMA, 1_000),  # negative spot
        (0.0, K, T, SIGMA, 1_000),  # zero spot
        (S0, -10.0, T, SIGMA, 1_000),  # negative strike
        (S0, K, -1.0, SIGMA, 1_000),  # negative maturity
        (S0, K, 0.0, SIGMA, 1_000),  # zero maturity
        (S0, K, T, -0.2, 1_000),  # negative volatility
        (S0, K, T, 0.0, 1_000),  # zero volatility
        (S0, K, T, SIGMA, 0),  # zero paths
        (S0, K, T, SIGMA, -100),  # negative paths
    ],
)
def test_invalid_inputs_raise_value_error(
    S: float, K_: float, T_: float, sigma: float, n_paths: int
) -> None:
    """Non-positive S, K, T, sigma or n_paths must raise ValueError."""
    with pytest.raises(ValueError):
        mc_price(S, K_, T_, R, sigma, n_paths=n_paths)


def test_invalid_option_type_raises_value_error() -> None:
    """An option_type other than 'call'/'put' must raise ValueError."""
    with pytest.raises(ValueError):
        mc_price(S0, K, T, R, SIGMA, option_type="straddle")
