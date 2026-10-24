"""Monte Carlo estimators of Greeks: pathwise, likelihood ratio and bump-and-reprice.

All three differentiate E[e^{-rT} f(S_T)] under GBM but in different places:

* Pathwise: differentiate the payoff, d f(S_T)/d theta. Unbiased and low
  variance, but needs a payoff that is (almost everywhere) differentiable, so
  it fails for digitals, whose derivative is zero almost everywhere.
* Likelihood ratio: differentiate the density of S_T instead, leaving the
  payoff untouched. Works for discontinuous payoffs, at the cost of higher
  variance.
* Bump-and-reprice: finite differences of the MC price. With common random
  numbers it is cheap and general; with independent draws the noise explodes
  as the bump shrinks (variance ~ 1/h^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pricing.black_scholes import validate_positive_inputs

_PAYOFFS = ("call", "put", "digital_call")


@dataclass(frozen=True)
class GreekEstimate:
    """A Monte Carlo Greek estimate with its standard error."""

    value: float
    std_error: float


def _terminal(S: float, T: float, r: float, sigma: float, q: float, z: np.ndarray) -> np.ndarray:
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _payoff(S_T: np.ndarray, K: float, payoff: str) -> np.ndarray:
    if payoff == "call":
        return np.maximum(S_T - K, 0.0)
    if payoff == "put":
        return np.maximum(K - S_T, 0.0)
    return (S_T > K).astype(float)


def _estimate(samples: np.ndarray) -> GreekEstimate:
    return GreekEstimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size)))


def mc_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payoff: str = "call",
    q: float = 0.0,
    method: str = "pathwise",
    n_paths: int = 200_000,
    bump: float = 0.01,
    common_random_numbers: bool = True,
    seed: int = 42,
) -> GreekEstimate:
    """Estimate delta by "pathwise", "likelihood_ratio" or "bump" (central difference).

    The bump is relative to S. Set common_random_numbers=False to see how
    badly independent draws behave in bump-and-reprice.

    Raises:
        ValueError: On an invalid payoff/method, or pathwise on a digital payoff.
    """
    validate_positive_inputs(S, K, T, sigma)
    if payoff not in _PAYOFFS:
        raise ValueError(f"payoff must be one of {_PAYOFFS}, got '{payoff}'.")
    if method not in ("pathwise", "likelihood_ratio", "bump"):
        raise ValueError(f"Unknown method '{method}'.")
    if method == "pathwise" and payoff == "digital_call":
        raise ValueError("Pathwise delta is invalid for a digital payoff (zero a.e. derivative).")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    disc = np.exp(-r * T)

    if method == "pathwise":
        S_T = _terminal(S, T, r, sigma, q, z)
        sign = 1.0 if payoff == "call" else -1.0
        in_money = (S_T > K) if payoff == "call" else (S_T < K)
        return _estimate(disc * sign * in_money * S_T / S)

    if method == "likelihood_ratio":
        S_T = _terminal(S, T, r, sigma, q, z)
        return _estimate(disc * _payoff(S_T, K, payoff) * z / (S * sigma * np.sqrt(T)))

    h = bump * S
    z_dn = z if common_random_numbers else rng.standard_normal(n_paths)
    up = disc * _payoff(_terminal(S + h, T, r, sigma, q, z), K, payoff)
    dn = disc * _payoff(_terminal(S - h, T, r, sigma, q, z_dn), K, payoff)
    return _estimate((up - dn) / (2.0 * h))


def mc_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 200_000,
    seed: int = 42,
) -> GreekEstimate:
    """Pathwise vega of a European call/put: d S_T / d sigma = S_T (sqrt(T) Z - sigma T)."""
    validate_positive_inputs(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")
    z = np.random.default_rng(seed).standard_normal(n_paths)
    S_T = _terminal(S, T, r, sigma, q, z)
    dS = S_T * (np.sqrt(T) * z - sigma * T)
    sign = 1.0 if option_type == "call" else -1.0
    in_money = (S_T > K) if option_type == "call" else (S_T < K)
    return _estimate(np.exp(-r * T) * sign * in_money * dS)
