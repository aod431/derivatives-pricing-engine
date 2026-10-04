"""Monte Carlo simulation for European option pricing under the risk-neutral measure.

Simulates the terminal price S_T under risk-neutral GBM dynamics, averages
the discounted payoff, and reports the Monte Carlo standard error and a 95%
confidence interval. Supports antithetic variates for variance reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for a Monte Carlo pricing estimate."""

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int


def _validate_inputs(S: float, K: float, T: float, sigma: float, n_paths: int) -> None:
    """Raise ValueError if any Monte Carlo input is out of its valid domain."""
    if S <= 0:
        raise ValueError(f"Spot price S must be positive, got {S}.")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}.")
    if T <= 0:
        raise ValueError(f"Time to maturity T must be positive, got {T}.")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}.")


def _simulate_terminal_price(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    antithetic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate S_T under the risk-neutral GBM, optionally using antithetic variates."""
    if antithetic:
        # Pair each draw Z with its mirror -Z; truncate if n_paths is odd.
        n_half = (n_paths + 1) // 2
        z_half = rng.standard_normal(n_half)
        z = np.concatenate([z_half, -z_half])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    antithetic: bool = False,
    seed: int = 42,
) -> MonteCarloResult:
    """Price a European call or put by Monte Carlo simulation.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of simulated paths. Defaults to 100,000.
        antithetic: If True, use antithetic variates (Z and -Z pairs) for
            variance reduction. Defaults to False.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A MonteCarloResult with the price estimate, standard error, and
        95% confidence interval bounds.

    Raises:
        ValueError: If S, K, T, sigma or n_paths are not positive, or
            option_type is not "call"/"put".
    """
    _validate_inputs(S, K, T, sigma, n_paths)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")

    rng = np.random.default_rng(seed)
    S_T = _simulate_terminal_price(S, T, r, sigma, q, n_paths, antithetic, rng)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return MonteCarloResult(
        price=price,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
    )
