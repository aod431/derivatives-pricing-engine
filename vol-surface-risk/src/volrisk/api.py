"""FastAPI service: surface inspection, option pricing and portfolio risk.

Surfaces are built from stored snapshots (never fetched on request, so latency
and behaviour do not depend on Yahoo) and cached per symbol. Environment:
VOLRISK_SOURCE=sample always serves the committed synthetic sample;
VOLRISK_ALLOW_SAMPLE=1 falls back to it only when no real snapshot exists;
VOLRISK_DATA_DIR points at another data directory.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from volrisk.data import SnapshotNotFound, load_chain, load_sample, prepare_chain
from volrisk.risk import PortfolioPricer, Position, bs_vec, risk_report
from volrisk.surface import VolSurface, build_surface

OptionType = Literal["call", "put"]


# ------------------------------------------------------------------- schemas


class PriceRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    option_type: OptionType
    strike: float = Field(gt=0)
    maturity: float = Field(gt=0, le=10, description="Time to expiry in years (ACT/365).")
    vol_override: float | None = Field(default=None, gt=0, le=5, description="Use this vol instead of the surface.")
    vol_shift: float = Field(default=0.0, ge=-0.5, le=0.5, description="Absolute shift added to the vol (0.01 = 1 pt).")


class PositionIn(BaseModel):
    option_type: OptionType
    strike: float = Field(gt=0)
    maturity: float = Field(gt=0, le=10)
    quantity: float = Field(description="Signed units; negative = short.")

    @field_validator("quantity")
    @classmethod
    def non_zero(cls, v: float) -> float:
        if v == 0 or not math.isfinite(v):
            raise ValueError("quantity must be a non-zero finite number")
        return v


class VarSettings(BaseModel):
    confidence: float = Field(default=0.99, gt=0.5, lt=1)
    n_sims: int = Field(default=20_000, ge=1_000, le=200_000)
    spot_vol: float | None = Field(default=None, gt=0, le=3)
    vol_sd: float = Field(default=0.01, gt=0, le=0.2)
    corr: float = Field(default=-0.7, gt=-1, lt=1)
    horizon_days: float = Field(default=1.0, gt=0, le=30)
    tail_df: float = Field(default=5.0, gt=2, le=100)
    seed: int = 42


class RiskRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    positions: list[PositionIn] = Field(min_length=1, max_length=500)
    spot_shifts: list[float] = Field(default=[-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10], min_length=1, max_length=25)
    vol_shifts: list[float] = Field(default=[-0.05, -0.02, 0.0, 0.02, 0.05], min_length=1, max_length=25)
    skew_shift: float = Field(default=0.10, ge=0, le=1, description="Skew tilt: vol change per unit log-moneyness.")
    var: VarSettings = VarSettings()

    @field_validator("spot_shifts")
    @classmethod
    def spot_range(cls, v: list[float]) -> list[float]:
        if any(s <= -1 or s > 5 for s in v):
            raise ValueError("spot shifts must lie in (-1, 5] as relative moves")
        return v


# ------------------------------------------------------------------ app state


class SurfaceStore:
    """Builds and caches one surface per symbol from stored snapshots."""

    def __init__(self, loader: Callable[[str], VolSurface] | None = None) -> None:
        self._loader = loader or self._default_loader
        self._cache: dict[str, VolSurface] = {}

    @staticmethod
    def _default_loader(symbol: str) -> VolSurface:
        if os.environ.get("VOLRISK_SOURCE", "snapshot") == "sample":
            return build_surface(prepare_chain(load_sample(symbol)))
        allow = os.environ.get("VOLRISK_ALLOW_SAMPLE", "0") == "1"
        return build_surface(prepare_chain(load_chain(symbol, allow_sample=allow)))

    def get(self, symbol: str) -> VolSurface:
        key = symbol.upper()
        if key not in self._cache:
            try:
                self._cache[key] = self._loader(key)
            except SnapshotNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"Surface for {key} could not be built: {exc}") from exc
        return self._cache[key]


def _clean(obj):
    """Make nested results JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _extrapolation_warnings(surface: VolSurface, maturities: list[float]) -> list[str]:
    lo, hi = float(surface.maturities[0]), float(surface.maturities[-1])
    out = []
    if any(t < lo for t in maturities):
        out.append(f"Some maturities are below the shortest calibrated expiry ({lo:.3f}y); vol is extrapolated flat.")
    if any(t > hi for t in maturities):
        out.append(f"Some maturities are beyond the longest calibrated expiry ({hi:.3f}y); vol is extrapolated flat.")
    return out


def create_app(store: SurfaceStore | None = None) -> FastAPI:
    app = FastAPI(title="volrisk", version="0.1.0",
                  description="Implied vol surface, pricing and risk on top of the pricing package.")
    store = store or SurfaceStore()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/surface/{symbol}")
    def surface(symbol: str) -> dict:
        return _clean(store.get(symbol).diagnostics())

    @app.post("/price")
    def price(req: PriceRequest) -> dict:
        surf = store.get(req.symbol)
        if req.vol_override is not None:
            sigma, source = req.vol_override, "override"
        else:
            sigma, source = float(surf.vol(req.strike, req.maturity)) + req.vol_shift, "surface"
        sigma = max(sigma, 1e-4)
        q = surf.carry(req.maturity)
        is_call = req.option_type == "call"
        p, delta, gamma, vega, theta = (
            float(x) for x in bs_vec(surf.spot, req.strike, req.maturity, surf.rate, sigma, q, is_call)
        )
        out = {
            "symbol": surf.symbol, "price": p, "vol": sigma, "vol_source": source,
            "forward": surf.forward(req.maturity), "spot": surf.spot,
            "greeks": {"delta": delta, "gamma": gamma, "vega_per_vol_pt": vega * 0.01, "theta_per_day": theta / 365.0},
            "warnings": _extrapolation_warnings(surf, [req.maturity]),
        }
        return _clean(out)

    @app.post("/risk")
    def risk(req: RiskRequest) -> dict:
        surf = store.get(req.symbol)
        try:
            pricer = PortfolioPricer(surf, [Position(p.option_type, p.strike, p.maturity, p.quantity)
                                            for p in req.positions])
            v = req.var
            report = risk_report(pricer, v.confidence, v.n_sims, v.spot_vol, v.vol_sd, v.corr,
                                 v.horizon_days, v.tail_df, v.seed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        grid = pricer.scenario_grid(req.spot_shifts, req.vol_shifts)
        skew_up = pricer.scenario_grid(req.spot_shifts, [0.0], vol_skew=req.skew_shift)[0.0]
        skew_dn = pricer.scenario_grid(req.spot_shifts, [0.0], vol_skew=-req.skew_shift)[0.0]
        out = {
            "symbol": surf.symbol,
            "value": pricer.base_value,
            "greeks": pricer.greeks(),
            "bucketed_vega": pricer.bucketed_vega(),
            "positions": pricer.table().to_dict(orient="records"),
            "scenarios": {
                "spot_shifts": req.spot_shifts, "vol_shifts": req.vol_shifts, "pnl": grid.to_numpy(),
                "skew_tilt": req.skew_shift,
                "skew_up_pnl": skew_up.to_numpy(), "skew_down_pnl": skew_dn.to_numpy(),
            },
            "var_es": report,
            "warnings": _extrapolation_warnings(surf, [p.maturity for p in req.positions]),
        }
        return _clean(out)

    return app


app = create_app()
