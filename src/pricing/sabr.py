"""SABR model: Hagan's implied-volatility expansion and calibration.

    dF = alpha F^beta dW1,   d alpha = nu alpha dW2,   d<W1, W2> = rho dt

Hagan et al. (2002) give a closed-form asymptotic expansion for the Black
(lognormal) implied volatility. It is an approximation: it is known to
degrade for very low strikes and long maturities, where it can imply negative
densities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SabrParams:
    """SABR parameters. beta is usually fixed (0 = normal-like, 1 = lognormal-like)."""

    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol(F: float, K: float | np.ndarray, T: float, p: SabrParams) -> float | np.ndarray:
    """Hagan's lognormal implied volatility for forward F and strike(s) K."""
    K = np.asarray(K, dtype=float)
    one_b = 1.0 - p.beta
    logfk = np.log(F / K)
    fk_mid = (F * K) ** (one_b / 2.0)

    z = (p.nu / p.alpha) * fk_mid * logfk
    # x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)); z/x(z) -> 1 as z -> 0.
    small = np.abs(z) < 1e-7
    z_safe = np.where(small, 1.0, z)
    x = np.log((np.sqrt(1.0 - 2.0 * p.rho * z_safe + z_safe**2) + z_safe - p.rho) / (1.0 - p.rho))
    z_over_x = np.where(small, 1.0, z_safe / x)

    denom = fk_mid * (1.0 + one_b**2 / 24.0 * logfk**2 + one_b**4 / 1920.0 * logfk**4)
    correction = 1.0 + (
        one_b**2 / 24.0 * p.alpha**2 / fk_mid**2
        + 0.25 * p.rho * p.beta * p.nu * p.alpha / fk_mid
        + (2.0 - 3.0 * p.rho**2) / 24.0 * p.nu**2
    ) * T
    vol = p.alpha / denom * z_over_x * correction
    return float(vol) if vol.ndim == 0 else vol


def calibrate_sabr(
    F: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    beta: float = 0.5,
) -> SabrParams:
    """Fit (alpha, rho, nu) to a smile with beta held fixed.

    beta is not identifiable from a single smile (it is almost collinear with
    rho), so the usual practice is to fix it a priori and calibrate the rest.
    alpha is seeded from the ATM vol, which pins it down almost exactly.

    Raises:
        ValueError: If inputs have mismatched shapes or too few points.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    if strikes.shape != market_vols.shape or strikes.size < 3:
        raise ValueError("Need at least 3 matching (strike, vol) points.")

    atm_vol = float(np.interp(F, strikes, market_vols))
    alpha0 = atm_vol * F ** (1.0 - beta)

    def residuals(x: np.ndarray) -> np.ndarray:
        alpha, rho, nu = x
        return sabr_implied_vol(F, strikes, T, SabrParams(alpha, beta, rho, nu)) - market_vols

    result = least_squares(
        residuals,
        x0=[alpha0, -0.2, 0.5],
        bounds=([1e-6, -0.999, 1e-6], [np.inf, 0.999, 10.0]),
    )
    alpha, rho, nu = result.x
    return SabrParams(float(alpha), beta, float(rho), float(nu))
