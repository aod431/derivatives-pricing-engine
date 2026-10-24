"""Discrete delta-hedging simulation.

A trader sells a European call at an implied vol sigma_imp and hedges with
Black-Scholes delta computed at sigma_imp, rebalancing n_rebalance times,
while the stock actually moves with realised vol sigma_real. The terminal
hedging P&L has two lessons:

* Its mean is driven by the gap between implied and realised variance: you
  earn roughly 0.5 * Gamma * S^2 * (sigma_imp^2 - sigma_real^2) per unit time.
* Its standard deviation falls like 1/sqrt(n_rebalance): continuous hedging
  is only a limit, and every discrete hedge leaves gamma noise.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pricing.black_scholes import black_scholes_price, d1_d2, validate_positive_inputs


def delta_hedge_pnl(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_imp: float,
    sigma_real: float,
    n_rebalance: int,
    n_paths: int = 10_000,
    mu: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Terminal P&L (at T) of a short call delta-hedged at implied vol, one value per path.

    The seller collects the BS premium at sigma_imp, holds delta shares financed
    at r, and pays the payoff at T. P&L is expressed in time-T money.

    Args:
        mu: Real-world drift of the stock. Defaults to r (the delta-hedged P&L
            is then free of the drift, so this mainly demonstrates that).
    """
    validate_positive_inputs(S, K, T, sigma_imp)
    if sigma_real <= 0 or n_rebalance < 1 or n_paths < 1:
        raise ValueError("sigma_real must be positive; n_rebalance and n_paths >= 1.")
    drift = r if mu is None else mu

    rng = np.random.default_rng(seed)
    dt = T / n_rebalance
    spot = np.full(n_paths, S, dtype=float)

    premium = black_scholes_price(S, K, T, r, sigma_imp, "call")
    d1, _ = d1_d2(S, K, T, r, sigma_imp, 0.0)
    delta = np.full(n_paths, norm.cdf(d1))
    cash = premium - delta * spot  # money-market account after buying the initial hedge

    for i in range(1, n_rebalance + 1):
        z = rng.standard_normal(n_paths)
        spot = spot * np.exp((drift - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)
        cash *= np.exp(r * dt)
        if i < n_rebalance:
            tau = T - i * dt
            new_d1 = (np.log(spot / K) + (r + 0.5 * sigma_imp**2) * tau) / (sigma_imp * np.sqrt(tau))
            new_delta = norm.cdf(new_d1)
            cash -= (new_delta - delta) * spot
            delta = new_delta

    return cash + delta * spot - np.maximum(spot - K, 0.0)
