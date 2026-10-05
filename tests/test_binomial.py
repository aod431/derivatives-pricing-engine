"""Tests validating src/pricing/binomial.py against Black-Scholes and known properties."""

import pytest

from pricing.black_scholes import black_scholes_price
from pricing.binomial import binomial_price

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_binomial_converges_to_black_scholes(option_type: str) -> None:
    """With enough steps, the European CRR tree must approach the closed-form price."""
    bs_price = black_scholes_price(S0, K, T, R, SIGMA, option_type=option_type, q=Q)
    tree_price = binomial_price(
        S0, K, T, R, SIGMA, option_type=option_type, q=Q,
        n_steps=1000, exercise_style="european",
    )
    assert tree_price == pytest.approx(bs_price, abs=0.01)


def test_american_price_is_never_below_european_price() -> None:
    """The extra early-exercise right can only help: American >= European, always."""
    american = binomial_price(S0, K, T, R, SIGMA, option_type="put", q=Q, n_steps=500, exercise_style="american")
    european = binomial_price(S0, K, T, R, SIGMA, option_type="put", q=Q, n_steps=500, exercise_style="european")
    assert american >= european


def test_deep_itm_american_put_equals_immediate_exercise() -> None:
    """Deep in-the-money, an American put's early-exercise premium makes it worth exactly K - S."""
    deep_itm_spot = 60.0
    american_put = binomial_price(
        deep_itm_spot, K, T, R, SIGMA, option_type="put", q=Q,
        n_steps=200, exercise_style="american",
    )
    assert american_put == pytest.approx(K - deep_itm_spot, abs=1e-6)


def test_american_call_equals_european_call_without_dividends() -> None:
    """With q=0, early exercise of a call is never optimal: American == European exactly."""
    american_call = binomial_price(
        S0, K, T, R, SIGMA, option_type="call", q=0.0, n_steps=500, exercise_style="american"
    )
    european_call = binomial_price(
        S0, K, T, R, SIGMA, option_type="call", q=0.0, n_steps=500, exercise_style="european"
    )
    assert american_call == pytest.approx(european_call, abs=1e-9)


def test_american_call_exceeds_european_call_with_dividends() -> None:
    """With q > 0, early exercise of a call can be optimal: American > European."""
    american_call = binomial_price(
        S0, K, T, R, SIGMA, option_type="call", q=Q, n_steps=500, exercise_style="american"
    )
    european_call = binomial_price(
        S0, K, T, R, SIGMA, option_type="call", q=Q, n_steps=500, exercise_style="european"
    )
    assert american_call >= european_call


@pytest.mark.parametrize(
    "S, K_, T_, sigma, n_steps",
    [
        (-10.0, K, T, SIGMA, 100),
        (S0, 0.0, T, SIGMA, 100),
        (S0, K, 0.0, SIGMA, 100),
        (S0, K, T, -0.1, 100),
        (S0, K, T, SIGMA, 0),
        (S0, K, T, SIGMA, -5),
    ],
)
def test_invalid_inputs_raise_value_error(S: float, K_: float, T_: float, sigma: float, n_steps: int) -> None:
    """Non-positive S, K, T, sigma or n_steps must raise ValueError."""
    with pytest.raises(ValueError):
        binomial_price(S, K_, T_, R, sigma, n_steps=n_steps)


def test_invalid_option_type_raises_value_error() -> None:
    """An option_type other than 'call'/'put' must raise ValueError."""
    with pytest.raises(ValueError):
        binomial_price(S0, K, T, R, SIGMA, option_type="straddle")


def test_invalid_exercise_style_raises_value_error() -> None:
    """An exercise_style other than 'european'/'american' must raise ValueError."""
    with pytest.raises(ValueError):
        binomial_price(S0, K, T, R, SIGMA, exercise_style="bermudan")
