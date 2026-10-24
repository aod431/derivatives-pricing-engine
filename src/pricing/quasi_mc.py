"""Quasi-Monte Carlo with scrambled Sobol sequences.

Low-discrepancy points fill the unit cube more evenly than pseudo-random ones,
so for smooth, low-effective-dimension integrands the error decays close to
1/N instead of 1/sqrt(N). Deterministic QMC has no built-in error bar, so
randomised QMC is used: R independent scrambles give R unbiased estimates
whose spread yields a standard error.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

from pricing.black_scholes import validate_option_type, validate_positive_inputs
from pricing.monte_carlo import MonteCarloResult


def qmc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    log2_paths: int = 14,
    n_replicates: int = 16,
    seed: int = 42,
) -> MonteCarloResult:
    """Price a European option with randomised Sobol QMC.

    Each of the n_replicates scrambles uses 2**log2_paths points (a power of
    two keeps the Sobol balance properties), for a total of
    n_replicates * 2**log2_paths payoff evaluations.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    if log2_paths < 1 or n_replicates < 2:
        raise ValueError("log2_paths must be >= 1 and n_replicates >= 2.")

    estimates = np.empty(n_replicates)
    for i in range(n_replicates):
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed + i)
        u = sampler.random_base2(m=log2_paths).ravel()
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
        estimates[i] = np.exp(-r * T) * payoff.mean()

    price = float(estimates.mean())
    se = float(estimates.std(ddof=1) / np.sqrt(n_replicates))
    return MonteCarloResult(price, se, price - 1.96 * se, price + 1.96 * se, n_replicates * 2**log2_paths)
