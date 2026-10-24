"""Raw SVI parametrisation of the implied-variance smile, and arbitrage checks.

Gatheral's raw SVI models total implied variance w(k) = sigma_imp(k)^2 * T as

    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + s^2) ),   k = ln(K / F)

A single slice is free of butterfly arbitrage iff the risk-neutral density is
non-negative, i.e. g(k) >= 0 (Gatheral & Jacquier 2014). Across maturities,
absence of calendar-spread arbitrage requires total variance to be
non-decreasing in T at every fixed log-moneyness k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SviParams:
    """Raw SVI parameters: a (level), b (angle), rho (skew), m (shift), s (smoothing)."""

    a: float
    b: float
    rho: float
    m: float
    s: float


def svi_total_variance(k: np.ndarray | float, p: SviParams) -> np.ndarray | float:
    """Total implied variance w(k) = sigma^2 T for log-moneyness k."""
    k = np.asarray(k, dtype=float)
    w = p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.s**2))
    return float(w) if w.ndim == 0 else w


def butterfly_g(k: np.ndarray, p: SviParams) -> np.ndarray:
    """Gatheral's g(k); the slice is free of butterfly arbitrage iff g(k) >= 0 for all k."""
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, p)
    root = np.sqrt((k - p.m) ** 2 + p.s**2)
    w1 = p.b * (p.rho + (k - p.m) / root)
    w2 = p.b * p.s**2 / root**3
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - w1**2 / 4.0 * (1.0 / w + 0.25) + w2 / 2.0


def has_butterfly_arbitrage(p: SviParams, k_grid: np.ndarray | None = None, tol: float = -1e-8) -> bool:
    """True if g(k) dips below tol anywhere on the grid (default k in [-1.5, 1.5])."""
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    return bool(np.min(butterfly_g(k_grid, p)) < tol)


def calendar_violations(
    slices: list[tuple[float, SviParams]], k_grid: np.ndarray | None = None, tol: float = -1e-10
) -> list[tuple[float, float]]:
    """Return the (T1, T2) pairs of adjacent slices where w(k, T2) < w(k, T1) for some k."""
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 201)
    ordered = sorted(slices, key=lambda x: x[0])
    bad = []
    for (t1, p1), (t2, p2) in zip(ordered[:-1], ordered[1:]):
        if np.min(svi_total_variance(k_grid, p2) - svi_total_variance(k_grid, p1)) < tol:
            bad.append((t1, t2))
    return bad


def fit_svi(
    k: np.ndarray,
    total_variance: np.ndarray,
    weights: np.ndarray | None = None,
    enforce_no_arbitrage: bool = True,
) -> SviParams:
    """Least-squares fit of raw SVI to one maturity slice.

    Uses several starting points because the objective is non-convex. With
    enforce_no_arbitrage=True a penalty on negative g(k) is added, which keeps
    the fitted slice butterfly-free (a soft constraint, not a guarantee: the
    result should still be checked with has_butterfly_arbitrage).

    Raises:
        ValueError: If inputs have mismatched shapes or fewer than 5 points.
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(total_variance, dtype=float)
    if k.shape != w_mkt.shape or k.size < 5:
        raise ValueError("Need at least 5 matching (k, total variance) points.")
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    k_pen = np.linspace(k.min() - 0.5, k.max() + 0.5, 100)

    def unpack(x: np.ndarray) -> SviParams:
        return SviParams(*x)

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        res = np.sqrt(wts) * (svi_total_variance(k, p) - w_mkt)
        if enforce_no_arbitrage:
            g = butterfly_g(k_pen, p)
            res = np.concatenate([res, 10.0 * np.minimum(g, 0.0)])
        return res

    w_min, w_max = float(w_mkt.min()), float(w_mkt.max())
    k_span = max(float(k.max() - k.min()), 0.1)
    lower = [-w_max, 1e-4, -0.999, k.min() - k_span, 1e-3]
    upper = [w_max, 10.0, 0.999, k.max() + k_span, 2.0]

    best, best_cost = None, np.inf
    for rho0 in (-0.7, -0.3, 0.0, 0.3):
        for s0 in (0.05, 0.2, 0.5):
            x0 = [0.5 * w_min, 0.5 * w_max / k_span, rho0, float(k[np.argmin(w_mkt)]), s0]
            x0 = np.clip(x0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
            sol = least_squares(residuals, x0, bounds=(lower, upper))
            if sol.cost < best_cost:
                best, best_cost = sol, sol.cost
    return unpack(best.x)
