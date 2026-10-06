"""Asian options: arithmetic (Monte Carlo + control variate) and geometric (closed-form).

The arithmetic-average Asian option has no closed form under Black-Scholes
(a sum of correlated lognormal variables is not itself lognormal), so it is
priced by Monte Carlo. The geometric-average Asian option DOES have a
closed form (the geometric average of lognormal variables is itself
lognormal), and is used here as a control variate: simulating the same
paths for both averages and correcting the arithmetic MC estimate by the
geometric MC estimate's known error removes most of the simulation noise,
because the two averages are highly correlated path by path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricing.black_scholes import validate_option_type, validate_positive_inputs


@dataclass(frozen=True)
class AsianMonteCarloResult:
    """Container for an arithmetic-average Asian Monte Carlo estimate."""

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int


def geometric_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_fixings: int,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Closed-form price of a discretely-monitored geometric-average Asian option.

    The geometric average of a GBM path sampled at n_fixings equally-spaced
    dates is itself lognormal, so pricing it reduces to a Black-Scholes-style
    formula with an adjusted ("effective") volatility and drift. As
    n_fixings -> infinity this converges to the classic continuous-monitoring
    Kemna-Vorst formula (effective volatility sigma/sqrt(3)).

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years (the last fixing date). Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        n_fixings: Number of equally-spaced averaging dates over (0, T].
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.

    Returns:
        The fair value of the geometric-average Asian option.

    Raises:
        ValueError: If S, K, T or sigma are not positive, n_fixings is not
            positive, or option_type is invalid.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    if n_fixings <= 0:
        raise ValueError(f"n_fixings must be positive, got {n_fixings}.")

    n = n_fixings
    t_bar = T * (n + 1) / (2 * n)
    adj_drift = (r - q - 0.5 * sigma**2) * t_bar
    var_total = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    vol_total = np.sqrt(var_total)

    expected_G = S * np.exp(adj_drift + 0.5 * var_total)

    d1 = (np.log(S / K) + adj_drift + var_total) / vol_total
    d2 = d1 - vol_total

    discount = np.exp(-r * T)
    if option_type == "call":
        price = discount * (expected_G * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount * (K * norm.cdf(-d2) - expected_G * norm.cdf(-d1))

    return float(price)


def _simulate_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_fixings: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate n_paths risk-neutral GBM paths observed at n_fixings equally-spaced dates.

    Returns an array of shape (n_paths, n_fixings) of underlying prices.
    """
    dt = T / n_fixings
    z = rng.standard_normal((n_paths, n_fixings))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    return S * np.exp(log_paths)


def asian_arithmetic_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_fixings: int,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    use_control_variate: bool = True,
    seed: int = 42,
) -> AsianMonteCarloResult:
    """Price an arithmetic-average Asian option by Monte Carlo.

    Optionally uses the closed-form geometric-average Asian price as a
    control variate: both averages are computed from the SAME simulated
    paths, so their difference has much lower variance than the arithmetic
    payoff alone (the two averages are highly correlated path by path).

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        n_fixings: Number of equally-spaced averaging dates over (0, T].
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of simulated paths. Defaults to 100,000.
        use_control_variate: If True, apply the geometric-Asian control
            variate correction. Defaults to True.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        An AsianMonteCarloResult with the price estimate, standard error,
        and 95% confidence interval bounds.

    Raises:
        ValueError: If S, K, T, sigma or n_paths are not positive,
            n_fixings is not positive, or option_type is invalid.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    if n_fixings <= 0:
        raise ValueError(f"n_fixings must be positive, got {n_fixings}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}.")

    rng = np.random.default_rng(seed)
    paths = _simulate_paths(S, T, r, sigma, q, n_fixings, n_paths, rng)

    arithmetic_avg = paths.mean(axis=1)
    geometric_avg = np.exp(np.log(paths).mean(axis=1))

    discount = np.exp(-r * T)
    if option_type == "call":
        arithmetic_payoffs = discount * np.maximum(arithmetic_avg - K, 0.0)
        geometric_payoffs = discount * np.maximum(geometric_avg - K, 0.0)
    else:
        arithmetic_payoffs = discount * np.maximum(K - arithmetic_avg, 0.0)
        geometric_payoffs = discount * np.maximum(K - geometric_avg, 0.0)

    if use_control_variate:
        geometric_closed_form = geometric_asian_price(
            S, K, T, r, sigma, n_fixings, option_type=option_type, q=q
        )
        adjusted = arithmetic_payoffs - geometric_payoffs + geometric_closed_form
    else:
        adjusted = arithmetic_payoffs

    price = float(np.mean(adjusted))
    std_error = float(np.std(adjusted, ddof=1) / np.sqrt(n_paths))
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return AsianMonteCarloResult(
        price=price,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
    )
