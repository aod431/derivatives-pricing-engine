"""Closed-form Black-Scholes pricing for European options.

Prices European calls and puts on an underlying following a geometric
Brownian motion, under the risk-neutral measure with a constant
continuous dividend yield q.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def validate_positive_inputs(S: float, K: float, T: float, sigma: float) -> None:
    """Raise ValueError if any Black-Scholes input is out of its valid domain.

    Shared by every pricer built on top of Black-Scholes (closed-form and
    Greeks), so the validity domain is defined in exactly one place.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}.")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}.")
    if T <= 0:
        raise ValueError(f"Time to maturity T must be positive, got {T}.")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}.")


def validate_option_type(option_type: str) -> None:
    """Raise ValueError if option_type is not 'call' or 'put'."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    """Compute the d1 and d2 terms used in the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Price a European call or put under the Black-Scholes model.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        option_type: Either "call" or "put".
        q: Continuous dividend yield of the underlying. Defaults to 0.0.

    Returns:
        The fair value of the option under the risk-neutral measure.

    Raises:
        ValueError: If S, K, T or sigma are not positive, or option_type
            is not "call"/"put".
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return float(price)
