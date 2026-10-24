"""Heston stochastic volatility model: semi-analytic pricing and Monte Carlo.

Dynamics under the risk-neutral measure:

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,    d<W1, W2> = rho dt

The semi-analytic price inverts the characteristic function of ln S_T with
the Gil-Pelaez formula. The characteristic function is written in the
"little Heston trap" form (Albrecher et al.), which keeps the complex
logarithm continuous in the integration variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pricing.black_scholes import validate_option_type, validate_positive_inputs
from pricing.monte_carlo import MonteCarloResult


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters: v0 (initial variance), kappa, theta, xi (vol of vol), rho."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0 or self.theta <= 0:
            raise ValueError("v0 and theta must be positive.")
        if self.kappa <= 0 or self.xi <= 0:
            raise ValueError("kappa and xi must be positive.")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}.")

    @property
    def feller_satisfied(self) -> bool:
        """True if 2*kappa*theta >= xi^2, i.e. the variance process cannot hit zero."""
        return 2.0 * self.kappa * self.theta >= self.xi**2


def _log_cf(u: np.ndarray, T: float, r: float, q: float, S: float, p: HestonParams) -> np.ndarray:
    """Characteristic function E[exp(i u ln S_T)] under Q (little-trap form)."""
    xi2 = p.xi**2
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta**2 + xi2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    exp_dT = np.exp(-d * T)
    C = (r - q) * iu * T + (p.kappa * p.theta / xi2) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = (beta - d) / xi2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(C + D * p.v0 + iu * np.log(S))


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
    q: float = 0.0,
    n_nodes: int = 400,
    u_max: float = 200.0,
) -> float:
    """Price a European option in the Heston model by Fourier inversion.

    P = S e^{-qT} P1 - K e^{-rT} P2 with P_j recovered from the characteristic
    function by Gil-Pelaez. The integral is evaluated with Gauss-Legendre on
    [0, u_max]; the integrand decays exponentially so the truncation is safe
    for maturities above a few days.

    Raises:
        ValueError: If inputs are out of domain.
    """
    validate_positive_inputs(S, K, T, 1.0)
    validate_option_type(option_type)

    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * u_max * (nodes + 1.0)
    w = 0.5 * u_max * weights

    cf = lambda x: _log_cf(x, T, r, q, S, params)  # noqa: E731
    forward = S * np.exp((r - q) * T)
    phi2 = cf(u)
    phi1 = cf(u - 1j) / forward  # characteristic function under the share measure
    e = np.exp(-1j * u * np.log(K))

    p1 = 0.5 + np.sum(w * np.real(e * phi1 / (1j * u))) / np.pi
    p2 = 0.5 + np.sum(w * np.real(e * phi2 / (1j * u))) / np.pi

    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    if option_type == "call":
        return float(call)
    # put-call parity
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))


def heston_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
) -> MonteCarloResult:
    """Price a European option in Heston by Euler simulation with full truncation.

    Full truncation (Lord et al.) replaces v by max(v, 0) in both the drift and
    the diffusion of the variance and of the log-spot, so negative variances
    never feed back into the dynamics. The scheme is biased: the bias shrinks
    as n_steps grows, on top of the usual 1/sqrt(n_paths) sampling error.
    """
    validate_positive_inputs(S, K, T, 1.0)
    validate_option_type(option_type)
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_c = np.sqrt(1.0 - params.rho**2)

    log_s = np.full(n_paths, np.log(S))
    v = np.full(n_paths, params.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = params.rho * z1 + rho_c * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_s += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
        v += params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * sqrt_dt * z2

    S_T = np.exp(log_s)
    payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
    disc = np.exp(-r * T) * payoff

    price = float(disc.mean())
    se = float(disc.std(ddof=1) / np.sqrt(n_paths))
    return MonteCarloResult(price, se, price - 1.96 * se, price + 1.96 * se, n_paths)
