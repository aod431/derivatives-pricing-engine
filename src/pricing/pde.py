"""Finite-difference (PDE) pricing of vanilla options with Crank-Nicolson.

Solves the Black-Scholes PDE in time-to-maturity tau and x = ln S:

    V_tau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V

on a uniform x grid, from the payoff at tau = 0 up to tau = T. Crank-Nicolson
is second order in time and space, but the payoff kink makes it oscillate; the
first two steps are therefore fully implicit (Rannacher smoothing). American
exercise is handled by projecting onto the payoff after every step, which is
simple and accurate to first order near the free boundary.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from pricing.binomial import validate_exercise_style
from pricing.black_scholes import validate_option_type, validate_positive_inputs


def pde_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    exercise_style: str = "european",
    n_space: int = 400,
    n_time: int = 400,
    n_std: float = 6.0,
) -> float:
    """Price a European or American vanilla option by Crank-Nicolson finite differences.

    Args:
        n_space: Number of space intervals on the log-spot grid.
        n_time: Number of time steps.
        n_std: Half-width of the grid in standard deviations of ln S_T.

    Raises:
        ValueError: If inputs are out of domain.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    validate_exercise_style(exercise_style)
    if n_space < 10 or n_time < 3:
        raise ValueError("n_space must be >= 10 and n_time >= 3.")

    american = exercise_style == "american"
    is_call = option_type == "call"
    mu = r - q - 0.5 * sigma**2

    centre = np.log(S) + mu * T
    half = n_std * sigma * np.sqrt(T)
    x = np.linspace(centre - half, centre + half, n_space + 1)
    dx = x[1] - x[0]
    spots = np.exp(x)
    dt = T / n_time

    payoff = np.maximum(spots - K, 0.0) if is_call else np.maximum(K - spots, 0.0)
    v = payoff.copy()

    lower = 0.5 * sigma**2 / dx**2 - mu / (2 * dx)
    diag = -(sigma**2) / dx**2 - r
    upper = 0.5 * sigma**2 / dx**2 + mu / (2 * dx)
    m = n_space - 1  # interior nodes

    def boundaries(tau: float) -> tuple[float, float]:
        if is_call:
            return 0.0, spots[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
        return K * np.exp(-r * tau) - spots[0] * np.exp(-q * tau), 0.0

    def apply_L(vec: np.ndarray) -> np.ndarray:
        return lower * vec[:-2] + diag * vec[1:-1] + upper * vec[2:]

    for step in range(n_time):
        theta = 1.0 if step < 2 else 0.5  # Rannacher start-up, then Crank-Nicolson
        tau_new = (step + 1) * dt

        ab = np.zeros((3, m))
        ab[0, 1:] = -theta * dt * upper
        ab[1, :] = 1.0 - theta * dt * diag
        ab[2, :-1] = -theta * dt * lower

        rhs = v[1:-1] + (1.0 - theta) * dt * apply_L(v)
        lo_b, hi_b = boundaries(tau_new)
        rhs[0] += theta * dt * lower * lo_b
        rhs[-1] += theta * dt * upper * hi_b

        v[1:-1] = solve_banded((1, 1), ab, rhs)
        v[0], v[-1] = lo_b, hi_b
        if american:
            v = np.maximum(v, payoff)

    return float(np.interp(np.log(S), x, v))
