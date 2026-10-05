"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Discretizes time into n_steps steps and approximates the risk-neutral GBM
with a recombining multiplicative random walk. The loop over time steps is
sequential (backward induction), but each step folds all of that step's
nodes at once with vectorized numpy operations instead of looping node by
node.
"""

from __future__ import annotations

import numpy as np

from pricing.black_scholes import validate_option_type, validate_positive_inputs


def validate_exercise_style(exercise_style: str) -> None:
    """Raise ValueError if exercise_style is not 'european' or 'american'."""
    if exercise_style not in ("european", "american"):
        raise ValueError(
            f"exercise_style must be 'european' or 'american', got '{exercise_style}'."
        )


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_steps: int = 500,
    exercise_style: str = "european",
) -> float:
    """Price a European or American option with a CRR binomial tree.

    Args:
        S: Current spot price of the underlying. Must be positive.
        K: Strike price. Must be positive.
        T: Time to maturity in years. Must be positive.
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility of the underlying. Must be positive.
        option_type: Either "call" or "put".
        q: Continuous dividend yield. Defaults to 0.0.
        n_steps: Number of time steps in the tree. Defaults to 500.
        exercise_style: "european" (exercise only at maturity) or
            "american" (exercise allowed at any node). Defaults to "european".

    Returns:
        The option's fair value under the CRR tree.

    Raises:
        ValueError: If S, K, T, sigma or n_steps are not positive, or
            option_type / exercise_style are invalid.
    """
    validate_positive_inputs(S, K, T, sigma)
    validate_option_type(option_type)
    validate_exercise_style(exercise_style)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal underlying prices at the n_steps+1 final nodes.
    j = np.arange(n_steps + 1)
    S_T = S * u**j * d ** (n_steps - j)

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction: fold each step's n+1 node values into n values.
    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])

        if exercise_style == "american":
            j = np.arange(step + 1)
            S_node = S * u**j * d ** (step - j)
            if option_type == "call":
                intrinsic = np.maximum(S_node - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_node, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
