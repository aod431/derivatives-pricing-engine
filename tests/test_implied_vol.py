"""Tests validating src/pricing/implied_vol.py against Black-Scholes itself."""

import numpy as np
import pytest

from pricing.black_scholes import black_scholes_price
from pricing.implied_vol import implied_volatility

S0, K, T, R, Q = 100.0, 100.0, 1.0, 0.05, 0.02


@pytest.mark.parametrize("true_sigma", [0.05, 0.10, 0.20, 0.35, 0.60, 1.50])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_volatility_at_the_money(true_sigma: float, option_type: str) -> None:
    """Pricing at a known sigma and inverting must recover that same sigma."""
    price = black_scholes_price(S0, K, T, R, true_sigma, option_type=option_type, q=Q)
    recovered = implied_volatility(price, S0, K, T, R, option_type=option_type, q=Q)
    assert recovered == pytest.approx(true_sigma, abs=1e-6)


@pytest.mark.parametrize("spot", [60.0, 85.0, 100.0, 115.0, 140.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_volatility_across_moneyness(spot: float, option_type: str) -> None:
    """Recovery must hold deep ITM and deep OTM, not just at the money."""
    true_sigma = 0.25
    price = black_scholes_price(spot, K, T, R, true_sigma, option_type=option_type, q=Q)
    recovered = implied_volatility(price, spot, K, T, R, option_type=option_type, q=Q)
    assert recovered == pytest.approx(true_sigma, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_below_lower_bound_raises_value_error(option_type: str) -> None:
    """A negative price is always below the no-arbitrage lower bound."""
    with pytest.raises(ValueError):
        implied_volatility(-5.0, S0, K, T, R, option_type=option_type, q=Q)


def test_call_price_above_spot_raises_value_error() -> None:
    """A call priced above S*exp(-qT) admits no finite volatility."""
    impossible_price = S0 * 2.0
    with pytest.raises(ValueError):
        implied_volatility(impossible_price, S0, K, T, R, option_type="call", q=Q)


def test_put_price_above_discounted_strike_raises_value_error() -> None:
    """A put priced above K*exp(-rT) admits no finite volatility."""
    impossible_price = K * np.exp(-R * T) + 10.0
    with pytest.raises(ValueError):
        implied_volatility(impossible_price, S0, K, T, R, option_type="put", q=Q)


@pytest.mark.parametrize(
    "S, K_, T_",
    [
        (-10.0, K, T),
        (0.0, K, T),
        (S0, -10.0, T),
        (S0, K, -1.0),
        (S0, K, 0.0),
    ],
)
def test_invalid_inputs_raise_value_error(S: float, K_: float, T_: float) -> None:
    """Non-positive S, K or T must raise ValueError regardless of price."""
    with pytest.raises(ValueError):
        implied_volatility(10.0, S, K_, T_, R)


def test_invalid_option_type_raises_value_error() -> None:
    """An option_type other than 'call'/'put' must raise ValueError."""
    with pytest.raises(ValueError):
        implied_volatility(10.0, S0, K, T, R, option_type="straddle")
