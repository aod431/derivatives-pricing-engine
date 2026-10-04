"""Implied volatility: inverting Black-Scholes to recover sigma from a price.

Given a European option's observed price, solves for the volatility sigma
that makes black_scholes_price(..., sigma) equal to that price. Uses Brent's
method, which only needs a bracketing interval and is guaranteed to converge
because option price is strictly increasing in sigma (Vega > 0 for sigma > 0),
unlike Newton-Raphson, which can diverge when Vega is very small (deep
ITM/OTM options, very short maturities).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pricing.black_scholes import black_scholes_price, validate_option_type


def _no_arbitrage_bounds(
    S: float, K: float, T: float, r: float, q: float, option_type: str
) -> tuple[float, float]:
    """Return the (lower, upper) bounds a price must lie within to imply a finite sigma.

    As sigma -> 0+, price converges to the discounted forward intrinsic value
    (the lower bound). As sigma -> infinity, a call converges to S*exp(-qT)
    and a put converges to K*exp(-rT) (the upper bound). A quoted price
    outside this open interval cannot correspond to any finite, positive
    volatility under Black-Scholes.
    """
    forward = S * np.exp((r - q) * T)
    if option_type == "call":
        lower = np.exp(-r * T) * max(forward - K, 0.0)
        upper = S * np.exp(-q * T)
    else:
        lower = np.exp(-r * T) * max(K - forward, 0.0)
        upper = K * np.exp(-r * T)
    return lower, upper


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    sigma_bracket: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Solve for the Black-Scholes volatility implied by a given option price.

    Args:
        price: Observed (or synthetic) option price. Must lie strictly
            within the model's no-arbitrage bounds for a solution to exist.
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.
        sigma_bracket: (low, high) volatility bracket passed to Brent's
            method. Must bracket the true implied volatility; the default
            (1e-6, 5.0) covers any realistic case.

    Returns:
        The implied volatility sigma such that
        black_scholes_price(S, K, T, r, sigma, option_type, q) == price.

    Raises:
        ValueError: If S, K or T are not positive, option_type is invalid,
            or price falls outside the model's no-arbitrage bounds (no
            finite volatility can reproduce it).
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}.")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}.")
    if T <= 0:
        raise ValueError(f"Time to maturity T must be positive, got {T}.")
    validate_option_type(option_type)

    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, option_type)
    if not (lower < price < upper):
        raise ValueError(
            f"Price {price} is outside the no-arbitrage bounds "
            f"({lower:.6f}, {upper:.6f}) for this option: no finite "
            "positive volatility can reproduce it."
        )

    def price_difference(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type=option_type, q=q) - price

    sigma_lo, sigma_hi = sigma_bracket
    return brentq(price_difference, sigma_lo, sigma_hi, xtol=1e-10)
