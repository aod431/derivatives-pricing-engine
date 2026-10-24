"""Longstaff-Schwartz Monte Carlo for American options.

Backward in time, the continuation value of holding the option is estimated by
regressing the realised discounted future cash flows on polynomials of the
current spot, using only in-the-money paths (where the exercise decision is
non-trivial). Exercise when intrinsic value exceeds the fitted continuation
value.

The estimator is biased *low*: the regression-based exercise rule is
sub-optimal, so the cash flows it generates are worth at most the true value
(up to Monte Carlo noise).
"""

from __future__ import annotations

import numpy as np

from pricing.black_scholes import validate_option_type, validate_positive_inputs
from pricing.monte_carlo import MonteCarloResult


def lsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    q: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 50,
    degree: int = 3,
    seed: int = 42,
) -> MonteCarloResult:
    """Price an American option (Bermudan on n_steps dates) with Longstaff-Schwartz.

    Args:
        n_steps: Number of exercise dates. The discrete-date price converges to the
            American price from below as n_steps grows.
        degree: Degree of the polynomial regression basis (in S / K).

    Raises:
        ValueError: If inputs are out of domain.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    if n_paths <= 0 or n_steps <= 0 or degree < 1:
        raise ValueError("n_paths, n_steps and degree must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Simulate the full grid of spots (columns = exercise dates 1..n_steps).
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = S * np.exp(np.cumsum(log_inc, axis=1))

    def intrinsic(s: np.ndarray) -> np.ndarray:
        return np.maximum(s - K, 0.0) if option_type == "call" else np.maximum(K - s, 0.0)

    # cash[i] = value (at the date of the last processed step) of the cash flow chosen for path i.
    cash = intrinsic(paths[:, -1])
    for t in range(n_steps - 2, -1, -1):
        cash *= disc
        s_t = paths[:, t]
        exercise_value = intrinsic(s_t)
        itm = exercise_value > 0.0
        if itm.sum() > degree + 1:
            x = s_t[itm] / K
            coeffs = np.polynomial.polynomial.polyfit(x, cash[itm], degree)
            continuation = np.polynomial.polynomial.polyval(x, coeffs)
            exercise_now = exercise_value[itm] > continuation
            idx = np.flatnonzero(itm)[exercise_now]
            cash[idx] = exercise_value[idx]

    samples = cash * disc
    # Exercising at t=0 is a deterministic comparison, so take the max with intrinsic value.
    price = float(max(samples.mean(), intrinsic(np.array(S))))
    se = float(samples.std(ddof=1) / np.sqrt(n_paths))
    return MonteCarloResult(price, se, price - 1.96 * se, price + 1.96 * se, n_paths)
