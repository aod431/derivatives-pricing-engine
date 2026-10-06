"""Tests validating src/pricing/exotics.py: geometric closed form and arithmetic MC."""

import pytest

from pricing.black_scholes import black_scholes_price
from pricing.exotics import asian_arithmetic_mc_price, geometric_asian_price

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
N_FIXINGS = 50


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_control_variate_price_consistent_with_plain_mc(option_type: str) -> None:
    """With or without the control variate, both estimators target the same true price."""
    plain = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type=option_type, q=Q,
        n_paths=50_000, use_control_variate=False, seed=42,
    )
    cv = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type=option_type, q=Q,
        n_paths=50_000, use_control_variate=True, seed=42,
    )
    assert abs(cv.price - plain.price) < 3 * plain.std_error


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_control_variate_reduces_standard_error(option_type: str) -> None:
    """The control variate must reduce the standard error, not just change the estimate."""
    plain = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type=option_type, q=Q,
        n_paths=50_000, use_control_variate=False, seed=42,
    )
    cv = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type=option_type, q=Q,
        n_paths=50_000, use_control_variate=True, seed=42,
    )
    assert cv.std_error < plain.std_error


def test_geometric_price_below_arithmetic_price_for_call() -> None:
    """Pathwise, the geometric average is never above the arithmetic average (AM-GM), and
    max(.) is monotonic, so a geometric Asian call can never be worth more than the
    arithmetic Asian call.
    """
    geometric_price = geometric_asian_price(S0, K, T, R, SIGMA, N_FIXINGS, option_type="call", q=Q)
    arithmetic_price = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type="call", q=Q, n_paths=100_000, seed=42
    ).price
    assert geometric_price < arithmetic_price


def test_geometric_price_above_arithmetic_price_for_put() -> None:
    """For a put, the AM-GM inequality flips direction: K - G >= K - A pointwise, so the
    geometric Asian put is worth at least as much as the arithmetic Asian put.
    """
    geometric_price = geometric_asian_price(S0, K, T, R, SIGMA, N_FIXINGS, option_type="put", q=Q)
    arithmetic_price = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type="put", q=Q, n_paths=100_000, seed=42
    ).price
    assert geometric_price > arithmetic_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_asian_cheaper_than_vanilla(option_type: str) -> None:
    """Averaging reduces the effective volatility of the payoff, so an Asian option must be
    cheaper than the vanilla European option with the same strike and maturity.
    """
    vanilla_price = black_scholes_price(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    asian_price = asian_arithmetic_mc_price(
        S0, K, T, R, SIGMA, N_FIXINGS, option_type=option_type, q=Q, n_paths=100_000, seed=42
    ).price
    assert asian_price < vanilla_price


@pytest.mark.parametrize(
    "S, K_, T_, sigma, n_fixings",
    [
        (-10.0, K, T, SIGMA, N_FIXINGS),
        (S0, 0.0, T, SIGMA, N_FIXINGS),
        (S0, K, 0.0, SIGMA, N_FIXINGS),
        (S0, K, T, -0.1, N_FIXINGS),
        (S0, K, T, SIGMA, 0),
        (S0, K, T, SIGMA, -5),
    ],
)
def test_geometric_invalid_inputs_raise_value_error(
    S: float, K_: float, T_: float, sigma: float, n_fixings: int
) -> None:
    """Non-positive S, K, T, sigma or n_fixings must raise ValueError."""
    with pytest.raises(ValueError):
        geometric_asian_price(S, K_, T_, R, sigma, n_fixings)


def test_geometric_invalid_option_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        geometric_asian_price(S0, K, T, R, SIGMA, N_FIXINGS, option_type="straddle")


@pytest.mark.parametrize(
    "S, K_, T_, sigma, n_fixings, n_paths",
    [
        (-10.0, K, T, SIGMA, N_FIXINGS, 1_000),
        (S0, 0.0, T, SIGMA, N_FIXINGS, 1_000),
        (S0, K, 0.0, SIGMA, N_FIXINGS, 1_000),
        (S0, K, T, -0.1, N_FIXINGS, 1_000),
        (S0, K, T, SIGMA, 0, 1_000),
        (S0, K, T, SIGMA, N_FIXINGS, 0),
    ],
)
def test_arithmetic_invalid_inputs_raise_value_error(
    S: float, K_: float, T_: float, sigma: float, n_fixings: int, n_paths: int
) -> None:
    """Non-positive S, K, T, sigma, n_fixings or n_paths must raise ValueError."""
    with pytest.raises(ValueError):
        asian_arithmetic_mc_price(S, K_, T_, R, sigma, n_fixings, n_paths=n_paths)


def test_arithmetic_invalid_option_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        asian_arithmetic_mc_price(S0, K, T, R, SIGMA, N_FIXINGS, option_type="straddle")
