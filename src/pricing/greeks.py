"""Black-Scholes Greeks: closed-form sensitivities and a finite-difference check.

Provides analytical delta, gamma, vega, theta and rho for a European call or
put, plus a generic central finite-difference estimator that can be used to
cross-check the analytical formulas against black_scholes_price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from pricing.black_scholes import d1_d2, validate_option_type, validate_positive_inputs


@dataclass(frozen=True)
class Greeks:
    """Container for the five standard Black-Scholes Greeks.

    delta, gamma and vega are expressed per unit change in S, S (twice) and
    sigma respectively. theta is per year (not per day) and rho is per unit
    change in r (not per 1% / per basis point).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def analytical_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> Greeks:
    """Compute closed-form Black-Scholes Greeks for a European call or put.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.

    Returns:
        A Greeks object with delta, gamma, vega, theta and rho.

    Raises:
        ValueError: If S, K, T or sigma are not positive, or option_type
            is not "call"/"put".
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    # Gamma and vega are identical for calls and puts: they come from the
    # shared N(d1) density term, which does not depend on option_type.
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def finite_difference_derivative(
    pricer: Callable[..., float],
    base_kwargs: dict,
    param_name: str,
    h: float,
    order: int = 1,
) -> float:
    """Estimate a derivative of pricer's output w.r.t. one of its keyword args.

    Uses a central difference for the first derivative (order=1) and a
    central second difference for the second derivative (order=2, used for
    gamma). Any pricing function that accepts keyword arguments can be
    passed in, which is what lets this same helper be reused later for
    models without closed-form Greeks (Monte Carlo, binomial, exotics).

    Args:
        pricer: A pricing function, e.g. black_scholes_price.
        base_kwargs: Keyword arguments to call pricer with.
        param_name: Name of the keyword argument to bump (e.g. "S", "sigma").
        h: Finite-difference step size applied to that parameter.
        order: 1 for the first derivative, 2 for the second derivative.

    Returns:
        The estimated derivative.

    Raises:
        ValueError: If order is not 1 or 2.
    """
    base_value = base_kwargs[param_name]

    up_kwargs = dict(base_kwargs)
    up_kwargs[param_name] = base_value + h
    down_kwargs = dict(base_kwargs)
    down_kwargs[param_name] = base_value - h

    price_up = pricer(**up_kwargs)
    price_down = pricer(**down_kwargs)

    if order == 1:
        return (price_up - price_down) / (2 * h)
    if order == 2:
        price_mid = pricer(**base_kwargs)
        return (price_up - 2 * price_mid + price_down) / (h**2)
    raise ValueError(f"order must be 1 or 2, got {order}.")
