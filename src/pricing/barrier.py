"""Single-barrier options: Reiner-Rubinstein closed form and Monte Carlo.

Closed forms assume continuous monitoring, zero rebate and a barrier that has
not been hit at inception. Knock-in prices come from the A-D building blocks
(Reiner & Rubinstein 1991, in Haug's notation); knock-out prices follow from
in-out parity: knock-in + knock-out = vanilla.

Monte Carlo simulates the log-spot on a discrete grid. Checking the barrier
only at grid dates misses crossings *between* dates and so overprices
knock-outs. The Brownian-bridge correction weights each path by the exact
probability of not crossing between consecutive dates, removing that bias.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pricing.black_scholes import black_scholes_price, validate_option_type, validate_positive_inputs
from pricing.monte_carlo import MonteCarloResult

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def _validate_barrier(S: float, H: float, barrier_type: str) -> None:
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}, got '{barrier_type}'.")
    if H <= 0:
        raise ValueError(f"Barrier H must be positive, got {H}.")
    if barrier_type.startswith("down") and S <= H:
        raise ValueError(f"Down barrier requires S > H, got S={S}, H={H}.")
    if barrier_type.startswith("up") and S >= H:
        raise ValueError(f"Up barrier requires S < H, got S={S}, H={H}.")


def barrier_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    q: float = 0.0,
) -> float:
    """Closed-form price of a continuously monitored single-barrier option (no rebate)."""
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    _validate_barrier(S, H, barrier_type)

    b = r - q
    sqt = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    phi = 1.0 if option_type == "call" else -1.0
    eta = 1.0 if barrier_type.startswith("down") else -1.0

    x1 = np.log(S / K) / sqt + (1 + mu) * sqt
    x2 = np.log(S / H) / sqt + (1 + mu) * sqt
    y1 = np.log(H**2 / (S * K)) / sqt + (1 + mu) * sqt
    y2 = np.log(H / S) / sqt + (1 + mu) * sqt

    fwd = S * np.exp((b - r) * T)
    disc = K * np.exp(-r * T)
    hs = H / S

    A = phi * fwd * norm.cdf(phi * x1) - phi * disc * norm.cdf(phi * x1 - phi * sqt)
    B = phi * fwd * norm.cdf(phi * x2) - phi * disc * norm.cdf(phi * x2 - phi * sqt)
    C = phi * fwd * hs ** (2 * (mu + 1)) * norm.cdf(eta * y1) - phi * disc * hs ** (2 * mu) * norm.cdf(
        eta * y1 - eta * sqt
    )
    D = phi * fwd * hs ** (2 * (mu + 1)) * norm.cdf(eta * y2) - phi * disc * hs ** (2 * mu) * norm.cdf(
        eta * y2 - eta * sqt
    )

    above = K > H
    if barrier_type == "down-and-in":
        knock_in = (C if above else A - B + D) if option_type == "call" else (B - C + D if above else A)
    elif barrier_type == "up-and-in":
        knock_in = (A if above else B - C + D) if option_type == "call" else (A - B + D if above else C)
    else:
        # Knock-out via in-out parity: build the matching knock-in, subtract from vanilla.
        in_type = barrier_type.replace("out", "in")
        knock_in = barrier_price(S, K, H, T, r, sigma, option_type, in_type, q)
        vanilla = black_scholes_price(S, K, T, r, sigma, option_type, q)
        return float(vanilla - knock_in)
    return float(knock_in)


def barrier_mc_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 50,
    brownian_bridge: bool = True,
    seed: int = 42,
) -> MonteCarloResult:
    """Monte Carlo price of a knock-out option (knock-ins via parity with the vanilla).

    With brownian_bridge=False the barrier is monitored only at the n_steps grid
    dates (the price of a *discretely* monitored option, which is higher for
    knock-outs). With brownian_bridge=True each path is weighted by the exact
    probability of staying on the right side of H between grid dates, which
    estimates the continuously monitored price.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    _validate_barrier(S, H, barrier_type)
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")

    is_in = barrier_type.endswith("in")
    is_down = barrier_type.startswith("down")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_h = np.log(H)
    x = np.full(n_paths, np.log(S))
    survival = np.ones(n_paths)
    alive = np.ones(n_paths, dtype=bool)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        x_new = x + (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        # Discrete monitoring: kill the path if the new date is across the barrier.
        crossed = (x_new <= log_h) if is_down else (x_new >= log_h)
        alive &= ~crossed
        if brownian_bridge:
            # P(no crossing between dates | both endpoints on the safe side).
            d0 = (x - log_h) if is_down else (log_h - x)
            d1 = (x_new - log_h) if is_down else (log_h - x_new)
            p_cross = np.exp(-2.0 * np.maximum(d0, 0.0) * np.maximum(d1, 0.0) / (sigma**2 * dt))
            survival *= 1.0 - p_cross
        x = x_new

    S_T = np.exp(x)
    payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
    weight = alive * (survival if brownian_bridge else 1.0)
    out_payoff = payoff * weight
    disc = np.exp(-r * T)

    if is_in:
        vanilla = payoff
        samples = disc * (vanilla - out_payoff)
    else:
        samples = disc * out_payoff

    price = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(n_paths))
    return MonteCarloResult(price, se, price - 1.96 * se, price + 1.96 * se, n_paths)
