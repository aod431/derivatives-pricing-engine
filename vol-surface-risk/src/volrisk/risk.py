"""Portfolio pricing and risk on top of a VolSurface.

Conventions
-----------
* Each option is priced with Black-Scholes at its own surface vol, vol(K, T),
  using the surface's carry q(T). Vol is sticky-strike: when spot moves the
  option keeps its strike-based vol, only the explicit shocks change it.
  Consequently the BS delta is the sticky-strike delta (no skew term).
* Quantity is in units of the underlying (a contract multiplier is the caller's
  business). Long = positive.
* Vega is quoted per one vol point (0.01 absolute), theta per calendar day.
* Scenario P&L is full revaluation minus current value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from volrisk.surface import VolSurface

DAYS_PER_YEAR = 365.0
MIN_VOL = 1e-4


@dataclass(frozen=True)
class Position:
    """A vanilla option position."""

    option_type: str
    strike: float
    T: float
    quantity: float

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'.")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")


def bs_vec(S, K, T, r, sigma, q, is_call):
    """Vectorised Black-Scholes price, delta, gamma, vega, theta (broadcasting over all args)."""
    sqt = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqt)
    d2 = d1 - sigma * sqt
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    sign = np.where(is_call, 1.0, -1.0)
    price = sign * (S * disc_q * ndtr(sign * d1) - K * disc_r * ndtr(sign * d2))
    delta = sign * disc_q * ndtr(sign * d1)
    gamma = disc_q * pdf / (S * sigma * sqt)
    vega = S * disc_q * pdf * sqt
    theta = (-S * disc_q * pdf * sigma / (2 * sqt)
             - sign * r * K * disc_r * ndtr(sign * d2) + sign * q * S * disc_q * ndtr(sign * d1))
    return price, delta, gamma, vega, theta


class PortfolioPricer:
    """Prices a portfolio under shocks to spot, vol (parallel / skew / per-position) and time."""

    def __init__(self, surface: VolSurface, positions: list[Position]) -> None:
        if not positions:
            raise ValueError("Portfolio is empty.")
        self.surface = surface
        self.positions = positions
        self.S = surface.spot
        self.r = surface.rate
        self.K = np.array([p.strike for p in positions])
        self.T = np.array([p.T for p in positions])
        self.qty = np.array([p.quantity for p in positions])
        self.is_call = np.array([p.option_type == "call" for p in positions])
        self.q = np.array([surface.carry(t) for t in self.T])
        self.vol0 = np.asarray(surface.vol(self.K, self.T), dtype=float)
        self.y0 = np.log(self.K / np.array([surface.forward(t) for t in self.T]))
        self.base_value = float(self.value())

    # ---- valuation ---------------------------------------------------------
    def value(self, spot_pct=0.0, vol_shift=0.0, vol_skew=0.0, dt_years=0.0):
        """Portfolio value; spot_pct / vol_shift / dt_years may be arrays of shape (n, 1) for scenarios.

        vol_shift is absolute (0.01 = one vol point) and may be a per-position array.
        Returns a float for scalar inputs, otherwise an array of values per scenario.
        """
        S = self.S * (1.0 + np.asarray(spot_pct, dtype=float))
        sigma = np.maximum(self.vol0 + vol_shift + vol_skew * self.y0, MIN_VOL)
        T = np.maximum(self.T - dt_years, 1e-6)
        price = bs_vec(S, self.K, T, self.r, sigma, self.q, self.is_call)[0]
        total = np.sum(self.qty * price, axis=-1)
        return float(total) if np.ndim(total) == 0 else total

    def table(self) -> pd.DataFrame:
        """Per-position price, vol and Greeks (vega per vol point, theta per day)."""
        price, delta, gamma, vega, theta = bs_vec(self.S, self.K, self.T, self.r, self.vol0, self.q, self.is_call)
        return pd.DataFrame({
            "option_type": [p.option_type for p in self.positions], "strike": self.K, "T": self.T,
            "quantity": self.qty, "vol": self.vol0, "price": price, "delta": delta, "gamma": gamma,
            "vega_pt": vega * 0.01, "theta_day": theta / DAYS_PER_YEAR, "value": self.qty * price,
        })

    def greeks(self) -> dict:
        t = self.table()
        q = t["quantity"]
        return {
            "value": float(t["value"].sum()),
            "delta": float((q * t["delta"]).sum()),
            "gamma": float((q * t["gamma"]).sum()),
            "vega": float((q * t["vega_pt"]).sum()),
            "theta": float((q * t["theta_day"]).sum()),
            "cash_delta": float((q * t["delta"]).sum() * self.S),
        }

    def bucketed_vega(self, bump: float = 0.01) -> dict[str, float]:
        """Vega per surface expiry, per vol point, by full-revaluation central difference.

        The bump is distributed over expiries with triangular weights centred on each
        slice (weights sum to 1 at any T), so the buckets add up to the parallel vega.
        """
        Ts = self.surface.maturities
        out = {}
        for j, sl in enumerate(self.surface.slices):
            basis = np.zeros(len(Ts))
            basis[j] = 1.0
            w_pos = np.interp(self.T, Ts, basis)  # flat beyond the ends
            up = self.value(vol_shift=0.5 * bump * w_pos)
            dn = self.value(vol_shift=-0.5 * bump * w_pos)
            out[sl.expiry] = float((up - dn) * (0.01 / bump))
        return out

    def scenario_grid(self, spot_shifts, vol_shifts, vol_skew: float = 0.0) -> pd.DataFrame:
        """P&L for every (spot shift, parallel vol shift) pair; rows = spot, columns = vol."""
        spot = np.asarray(spot_shifts, dtype=float)
        vol = np.asarray(vol_shifts, dtype=float)
        pnl = np.empty((len(spot), len(vol)))
        for j, v in enumerate(vol):
            pnl[:, j] = self.value(spot_pct=spot[:, None], vol_shift=v, vol_skew=vol_skew) - self.base_value
        return pd.DataFrame(pnl, index=spot, columns=vol)


# ----------------------------------------------------------------- VaR / ES


def simulate_risk_factors(
    n: int, spot_vol: float, vol_sd: float, corr: float = -0.7, horizon_days: float = 1.0,
    tail_df: float = 5.0, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint draws of (relative spot move, absolute parallel vol change) over the horizon.

    Fat tails: a bivariate normal is multiplied by sqrt((df-2)/chi2_df), a multivariate
    Student-t with unit variance. spot_vol is annualised; vol_sd is the stdev of the
    vol change over the horizon (0.01 = one vol point).
    """
    if not -1 < corr < 1:
        raise ValueError("corr must be in (-1, 1).")
    if tail_df <= 2:
        raise ValueError("tail_df must exceed 2 for a finite variance.")
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = corr * z1 + math.sqrt(1 - corr**2) * rng.standard_normal(n)
    scale = np.sqrt((tail_df - 2.0) / rng.chisquare(tail_df, n))
    h = horizon_days / 252.0
    return spot_vol * math.sqrt(h) * z1 * scale, vol_sd * z2 * scale


def var_es(pnl: np.ndarray, confidence: float = 0.99) -> tuple[float, float]:
    """Historical-style VaR and Expected Shortfall of a P&L sample, as positive loss numbers."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be in (0.5, 1).")
    cut = np.quantile(pnl, 1.0 - confidence)
    tail = pnl[pnl <= cut]
    return float(-cut), float(-tail.mean())


def pnl_full_revaluation(pricer: PortfolioPricer, spot_ret: np.ndarray, dvol: np.ndarray,
                         horizon_days: float = 1.0) -> np.ndarray:
    """Exact P&L per scenario: reprice every option at the shocked spot, vol and shorter maturity."""
    dt = horizon_days / DAYS_PER_YEAR
    values = pricer.value(spot_pct=spot_ret[:, None], vol_shift=dvol[:, None], dt_years=dt)
    return values - pricer.base_value


def pnl_delta_gamma_vega(pricer: PortfolioPricer, spot_ret: np.ndarray, dvol: np.ndarray,
                         horizon_days: float = 1.0) -> np.ndarray:
    """Second-order Taylor P&L: delta*dS + gamma*dS^2/2 + vega*dsigma + theta*dt (ignores vanna/volga)."""
    g = pricer.greeks()
    dS = pricer.S * spot_ret
    return (g["delta"] * dS + 0.5 * g["gamma"] * dS**2 + g["vega"] * dvol / 0.01
            + g["theta"] * horizon_days)


def risk_report(
    pricer: PortfolioPricer, confidence: float = 0.99, n_sims: int = 20_000, spot_vol: float | None = None,
    vol_sd: float = 0.01, corr: float = -0.7, horizon_days: float = 1.0, tail_df: float = 5.0, seed: int = 42,
) -> dict:
    """1-day Monte Carlo VaR/ES by full revaluation and by delta-gamma-vega, side by side.

    spot_vol defaults to the surface ATM vol at ~1 month.
    """
    surf = pricer.surface
    if spot_vol is None:
        T1 = float(np.clip(30 / 365, surf.maturities[0], surf.maturities[-1]))
        spot_vol = float(surf.vol(surf.forward(T1), T1))
    ret, dvol = simulate_risk_factors(n_sims, spot_vol, vol_sd, corr, horizon_days, tail_df, seed)
    full = pnl_full_revaluation(pricer, ret, dvol, horizon_days)
    approx = pnl_delta_gamma_vega(pricer, ret, dvol, horizon_days)
    v_full, e_full = var_es(full, confidence)
    v_dgv, e_dgv = var_es(approx, confidence)
    return {
        "confidence": confidence, "horizon_days": horizon_days, "n_sims": n_sims,
        "spot_vol": spot_vol, "vol_sd": vol_sd, "corr": corr, "tail_df": tail_df,
        "full_revaluation": {"var": v_full, "es": e_full, "mean_pnl": float(full.mean())},
        "delta_gamma_vega": {"var": v_dgv, "es": e_dgv, "mean_pnl": float(approx.mean())},
        "approx_error_pct_var": float(100 * (v_dgv - v_full) / v_full) if v_full else 0.0,
    }
