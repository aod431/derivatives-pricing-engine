"""Tests validating src/pricing/black_scholes.py against known analytical properties."""

import numpy as np
import pytest

from pricing.black_scholes import black_scholes_price

# Shared baseline parameters, consistent with the Black-Scholes notebook.
S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02


def test_put_call_parity() -> None:
    """C - P must equal S*exp(-qT) - K*exp(-rT) exactly, up to floating point error."""
    call = black_scholes_price(S0, K, T, R, SIGMA, option_type="call", q=Q)
    put = black_scholes_price(S0, K, T, R, SIGMA, option_type="put", q=Q)
    lhs = call - put
    rhs = S0 * np.exp(-Q * T) - K * np.exp(-R * T)
    assert np.isclose(lhs, rhs, atol=1e-10)


def test_small_maturity_converges_to_intrinsic_value() -> None:
    """As T -> 0, an in-the-money call converges to its discounted intrinsic payoff.

    Uses an in-the-money spot (not at-the-money) because at-the-money time
    value decays as sqrt(T), not T, and would converge far too slowly here.
    """
    itm_spot = 120.0
    tiny_T = 1e-6
    call = black_scholes_price(itm_spot, K, tiny_T, R, SIGMA, option_type="call", q=Q)
    intrinsic = max(itm_spot * np.exp(-Q * tiny_T) - K * np.exp(-R * tiny_T), 0.0)
    assert np.isclose(call, intrinsic, atol=1e-3)


def test_small_volatility_converges_to_discounted_forward_intrinsic() -> None:
    """As sigma -> 0, the price converges to the deterministic discounted forward payoff."""
    tiny_sigma = 1e-4
    call = black_scholes_price(S0, K, T, R, tiny_sigma, option_type="call", q=Q)
    forward = S0 * np.exp((R - Q) * T)
    expected = np.exp(-R * T) * max(forward - K, 0.0)
    assert np.isclose(call, expected, atol=1e-3)


def test_deep_itm_call_converges_to_forward_intrinsic() -> None:
    """A deep in-the-money call is worth its discounted forward intrinsic value."""
    deep_itm_spot = 1000.0
    call = black_scholes_price(deep_itm_spot, K, T, R, SIGMA, option_type="call", q=Q)
    expected = deep_itm_spot * np.exp(-Q * T) - K * np.exp(-R * T)
    assert np.isclose(call, expected, atol=1e-6)


def test_deep_otm_call_is_near_zero() -> None:
    """A deep out-of-the-money call is worth close to zero."""
    deep_otm_spot = 1.0
    call = black_scholes_price(deep_otm_spot, K, T, R, SIGMA, option_type="call", q=Q)
    assert call < 1e-6


@pytest.mark.parametrize(
    "S, K_, T_, sigma",
    [
        (-10.0, K, T, SIGMA),  # negative spot
        (0.0, K, T, SIGMA),  # zero spot
        (S0, -10.0, T, SIGMA),  # negative strike
        (S0, K, -1.0, SIGMA),  # negative maturity
        (S0, K, 0.0, SIGMA),  # zero maturity
        (S0, K, T, -0.2),  # negative volatility
        (S0, K, T, 0.0),  # zero volatility
    ],
)
def test_invalid_inputs_raise_value_error(S: float, K_: float, T_: float, sigma: float) -> None:
    """Non-positive S, K, T or sigma must raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_price(S, K_, T_, R, sigma)


def test_invalid_option_type_raises_value_error() -> None:
    """An option_type other than 'call'/'put' must raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_price(S0, K, T, R, SIGMA, option_type="straddle")
