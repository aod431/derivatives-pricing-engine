"""Implied volatility surface: per-expiry SVI slices, no-arbitrage repair, Dupire local vol.

Method
------
1. Invert Black-Scholes on every OTM quote, using that expiry's parity forward
   (spot S and carry q chosen so that S e^{(r-q)T} = F).
2. Fit raw SVI to total variance w = sigma^2 T against log-moneyness
   k = ln(K/F), one slice per expiry, weighting each quote by the inverse
   variance implied by its bid-ask width.
3. Check butterfly (g(k) >= 0) and calendar (w non-decreasing in T) arbitrage.
   Slices that fail are refit with penalty terms on the violated constraint.
   These are soft constraints: the result is re-checked and reported, never
   assumed clean.
4. Between slices total variance is interpolated linearly in T at fixed k,
   which preserves the absence of calendar arbitrage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pricing.greeks import analytical_greeks
from pricing.implied_vol import implied_volatility
from pricing.svi import (
    SviParams,
    butterfly_g,
    fit_svi,
    has_butterfly_arbitrage,
    svi_total_variance,
)

from volrisk.data import PreparedChain

BUTTERFLY_GRID = np.linspace(-1.5, 1.5, 601)
MIN_POINTS_PER_SLICE = 6


@dataclass
class SmileSlice:
    """One calibrated expiry."""

    expiry: str
    T: float
    forward: float
    implied_q: float
    params: SviParams
    n_quotes: int
    rmse_vol_pts: float
    k_min: float
    k_max: float
    repaired: bool = False
    butterfly_free: bool = True
    min_g: float = 0.0
    calendar_ok_to_next: bool | None = None  # None for the last slice


@dataclass
class VolSurface:
    """Calibrated surface for one underlying at one date."""

    symbol: str
    as_of: str
    spot: float
    rate: float
    slices: list[SmileSlice]
    market: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)  # quotes with market iv
    source: str = "snapshot"

    # ---- term structure -------------------------------------------------
    @property
    def maturities(self) -> np.ndarray:
        return np.array([s.T for s in self.slices])

    def carry(self, T: float) -> float:
        """Implied dividend/carry yield q(T): linear between slices, flat outside."""
        return float(np.interp(T, self.maturities, [s.implied_q for s in self.slices]))

    def forward(self, T: float) -> float:
        return self.spot * math.exp((self.rate - self.carry(T)) * T)

    # ---- total variance and vol -----------------------------------------
    def total_variance(self, y: np.ndarray | float, T: float) -> np.ndarray:
        """w(y, T) at log-forward-moneyness y.

        Linear in T between slices; outside the quoted range the vol at fixed y is
        held flat (w scales with T from the nearest slice).
        """
        y = np.asarray(y, dtype=float)
        Ts = self.maturities
        if T <= Ts[0]:
            return np.asarray(svi_total_variance(y, self.slices[0].params)) * T / Ts[0]
        if T >= Ts[-1]:
            return np.asarray(svi_total_variance(y, self.slices[-1].params)) * T / Ts[-1]
        j = int(np.searchsorted(Ts, T))
        lam = (T - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
        w0 = svi_total_variance(y, self.slices[j - 1].params)
        w1 = svi_total_variance(y, self.slices[j].params)
        return np.asarray((1.0 - lam) * w0 + lam * w1)

    def vol(self, K: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        """Black implied vol at strike K and maturity T (scalars or broadcastable arrays)."""
        K_arr, T_arr = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
        out = np.empty(K_arr.shape)
        for t in np.unique(T_arr):
            if t <= 0:
                raise ValueError("Maturity must be positive.")
            mask = T_arr == t
            y = np.log(K_arr[mask] / self.forward(float(t)))
            out[mask] = np.sqrt(np.maximum(self.total_variance(y, float(t)), 1e-12) / t)
        return float(out) if out.ndim == 0 else out

    def local_vol(self, K: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        """Dupire local volatility from the implied surface (Gatheral's total-variance form).

            sigma_loc^2 = dw/dT / [ 1 - (y/w) w_y + (1/4)(-1/4 - 1/w + y^2/w^2) w_y^2 + (1/2) w_yy ]

        Derivatives are central finite differences of the interpolated surface.
        Returns NaN where the denominator or dw/dT is not positive, i.e. where the
        surface admits no valid local vol (a numerical symptom of arbitrage).
        """
        K_arr, T_arr = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
        out = np.full(K_arr.shape, np.nan)
        hy = 5e-3
        for idx in np.ndindex(K_arr.shape):
            T0 = float(T_arr[idx])
            y = math.log(K_arr[idx] / self.forward(T0))
            hT = min(5e-3, 0.5 * T0)
            w = float(self.total_variance(y, T0))
            w_up, w_dn = float(self.total_variance(y + hy, T0)), float(self.total_variance(y - hy, T0))
            wy = (w_up - w_dn) / (2 * hy)
            wyy = (w_up - 2 * w + w_dn) / hy**2
            wT = (float(self.total_variance(y, T0 + hT)) - float(self.total_variance(y, T0 - hT))) / (2 * hT)
            denom = 1 - y / w * wy + 0.25 * (-0.25 - 1 / w + y**2 / w**2) * wy**2 + 0.5 * wyy
            if denom > 0 and wT > 0 and w > 0:
                out[idx] = math.sqrt(wT / denom)
        return float(out) if out.ndim == 0 else out

    # ---- diagnostics ------------------------------------------------------
    def diagnostics(self) -> dict:
        """Per-slice fit quality and arbitrage status, plus local-vol validity on a grid."""
        Ts = self.maturities
        ks = np.linspace(-0.2, 0.2, 21)
        T_grid = np.linspace(Ts[0], Ts[-1], 25)
        lv = np.array([[self.local_vol(self.forward(t) * math.exp(k), t) for k in ks] for t in T_grid])
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "spot": self.spot,
            "rate": self.rate,
            "source": self.source,
            "slices": [
                {
                    "expiry": s.expiry, "T": s.T, "forward": s.forward, "implied_q": s.implied_q,
                    "svi": {n: float(getattr(s.params, n)) for n in ("a", "b", "rho", "m", "s")},
                    "n_quotes": s.n_quotes, "rmse_vol_pts": float(s.rmse_vol_pts),
                    "butterfly_free": bool(s.butterfly_free), "min_g": float(s.min_g),
                    "calendar_ok_to_next": s.calendar_ok_to_next, "repaired": s.repaired,
                }
                for s in self.slices
            ],
            "local_vol_invalid_pct": float(100.0 * np.isnan(lv).mean()),
            "arbitrage_free": all(s.butterfly_free for s in self.slices)
            and all(s.calendar_ok_to_next in (True, None) for s in self.slices),
        }

    @classmethod
    def from_slices(cls, symbol: str, spot: float, rate: float, slices: list[SmileSlice], as_of: str = "",
                    source: str = "manual") -> "VolSurface":
        """Build a surface directly from slices (used for synthetic tests)."""
        return cls(symbol, as_of, spot, rate, sorted(slices, key=lambda s: s.T), pd.DataFrame(), source)


# ------------------------------------------------------------------ calibration


def implied_vols_from_quotes(prepared: PreparedChain) -> pd.DataFrame:
    """Black-Scholes implied vol for every OTM quote, with a bid-ask based weight.

    The weight is the inverse variance of total variance: the quote's half-width in
    vol terms is (ask - bid) / (2 vega), and dw = 2 sigma T d(sigma).
    """
    fw = prepared.forwards.set_index("expiry")
    rows = []
    for rec in prepared.otm.itertuples(index=False):
        q = float(fw.loc[rec.expiry, "implied_q"])
        try:
            iv = implied_volatility(rec.mid, prepared.spot, rec.strike, rec.T, prepared.rate, rec.option_type, q)
            vega = analytical_greeks(prepared.spot, rec.strike, rec.T, prepared.rate, iv, rec.option_type, q).vega
        except ValueError:
            continue  # price outside no-arbitrage bounds: no finite vol
        if not 0.01 < iv < 3.0 or vega <= 1e-8:
            continue
        iv_halfwidth = max((rec.ask - rec.bid) / (2.0 * vega), 1e-4)
        sd_w = 2.0 * iv * rec.T * iv_halfwidth
        rows.append((*rec, iv, iv * iv * rec.T, 1.0 / sd_w**2))
    cols = list(prepared.otm.columns) + ["iv", "w", "weight"]
    return pd.DataFrame(rows, columns=cols)


def _constrained_fit(
    k: np.ndarray,
    w: np.ndarray,
    wts: np.ndarray,
    x0: np.ndarray,
    w_prev: SviParams | None,
    k_cal: np.ndarray,
    penalty: float = 1e5,
) -> SviParams:
    """Refit one slice with penalties on butterfly (full grid) and calendar (vs previous slice)."""
    span = max(float(k.max() - k.min()), 0.1)
    lower = [-float(w.max()), 1e-4, -0.999, k.min() - span, 1e-3]
    upper = [float(w.max()), 10.0, 0.999, k.max() + span, 2.0]

    def residuals(x: np.ndarray) -> np.ndarray:
        p = SviParams(*x)
        res = [np.sqrt(wts) * (svi_total_variance(k, p) - w)]
        res.append(penalty * np.minimum(butterfly_g(BUTTERFLY_GRID, p), 0.0))
        if w_prev is not None:
            res.append(penalty * np.minimum(svi_total_variance(k_cal, p) - svi_total_variance(k_cal, w_prev), 0.0))
        return np.concatenate(res)

    x0 = np.clip(x0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
    fit = SviParams(*least_squares(residuals, x0, bounds=(lower, upper)).x)
    if w_prev is not None:
        # The penalty is soft; finish with the smallest parallel lift of w that removes any residual crossing.
        deficit = -float(np.min(svi_total_variance(k_cal, fit) - svi_total_variance(k_cal, w_prev)))
        if deficit > 0.0:
            fit = SviParams(fit.a + deficit + 1e-9, fit.b, fit.rho, fit.m, fit.s)
    return fit


def _calendar_ok(prev: SviParams, cur: SviParams, k_grid: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.min(svi_total_variance(k_grid, cur) - svi_total_variance(k_grid, prev)) >= -tol)


def fit_slices(
    ivs: pd.DataFrame, forwards: pd.DataFrame, repair: bool = True
) -> list[SmileSlice]:
    """Fit SVI per expiry (ascending T) and repair butterfly / calendar violations."""
    slices: list[SmileSlice] = []
    prev: SviParams | None = None
    for fw in forwards.sort_values("T").itertuples(index=False):
        grp = ivs[ivs.expiry == fw.expiry]
        if len(grp) < MIN_POINTS_PER_SLICE:
            continue
        k = grp["k"].to_numpy()
        w = grp["w"].to_numpy()
        wts = grp["weight"].to_numpy()
        wts = wts / wts.mean()

        params = fit_svi(k, w, wts, enforce_no_arbitrage=True)
        k_cal = np.linspace(k.min() - 0.1, k.max() + 0.1, 121)
        repaired = False
        bad_fly = has_butterfly_arbitrage(params, BUTTERFLY_GRID)
        bad_cal = prev is not None and not _calendar_ok(prev, params, k_cal)
        if repair and (bad_fly or bad_cal):
            params = _constrained_fit(k, w, wts, np.array([params.a, params.b, params.rho, params.m, params.s]),
                                      prev, k_cal)
            repaired = True

        fit_iv = np.sqrt(np.maximum(svi_total_variance(k, params), 1e-12) / fw.T)
        rmse = float(100.0 * np.sqrt(np.mean((fit_iv - grp["iv"].to_numpy()) ** 2)))
        g_min = float(np.min(butterfly_g(BUTTERFLY_GRID, params)))
        slices.append(SmileSlice(
            expiry=fw.expiry, T=float(fw.T), forward=float(fw.forward), implied_q=float(fw.implied_q),
            params=params, n_quotes=len(grp), rmse_vol_pts=rmse, k_min=float(k.min()), k_max=float(k.max()),
            repaired=repaired, butterfly_free=not has_butterfly_arbitrage(params, BUTTERFLY_GRID),
            min_g=g_min,
        ))
        prev = params

    for a, b in zip(slices[:-1], slices[1:]):
        k_cal = np.linspace(min(a.k_min, b.k_min) - 0.1, max(a.k_max, b.k_max) + 0.1, 121)
        a.calendar_ok_to_next = _calendar_ok(a.params, b.params, k_cal)
    return slices


def build_surface(prepared: PreparedChain, repair: bool = True) -> VolSurface:
    """Full calibration from a prepared chain.

    Raises:
        ValueError: If fewer than two expiries have enough quotes to calibrate.
    """
    ivs = implied_vols_from_quotes(prepared)
    slices = fit_slices(ivs, prepared.forwards, repair=repair)
    if len(slices) < 2:
        raise ValueError(f"Need at least 2 calibratable expiries, got {len(slices)}.")
    return VolSurface(prepared.symbol, prepared.as_of.isoformat(), prepared.spot, prepared.rate, slices,
                      ivs, prepared.source)
