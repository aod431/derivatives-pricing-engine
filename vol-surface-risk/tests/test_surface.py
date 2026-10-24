"""Surface: fit quality against the known Heston truth, arbitrage status, repair, local vol."""

import math

import numpy as np
import pytest

from pricing.heston import HestonParams, heston_price
from pricing.implied_vol import implied_volatility
from pricing.svi import SviParams, has_butterfly_arbitrage, svi_total_variance
from volrisk.surface import (
    BUTTERFLY_GRID, SmileSlice, VolSurface, _calendar_ok, build_surface, fit_slices, implied_vols_from_quotes,
)

HESTON = HestonParams(v0=0.035, kappa=2.0, theta=0.045, xi=0.55, rho=-0.75)


def test_fit_rmse_is_small_in_vol_points(surface):
    assert max(s.rmse_vol_pts for s in surface.slices) < 0.25  # quarter of a vol point
    assert len(surface.slices) == 5


def test_surface_has_no_butterfly_or_calendar_arbitrage(surface):
    d = surface.diagnostics()
    assert d["arbitrage_free"]
    for s in surface.slices:
        assert not has_butterfly_arbitrage(s.params, BUTTERFLY_GRID)


def test_surface_matches_true_heston_vols_inside_the_data(surface, chain):
    """Compare to Heston implied vols at points that were not quoted (off-grid strikes, mid maturities)."""
    for T, K_over_F in [(0.25, 0.97), (0.5, 1.03), (0.5, 0.92), (1.0, 1.05)]:
        F = surface.forward(T)
        K = F * K_over_F
        q = surface.carry(T)
        true_iv = implied_volatility(heston_price(chain.spot, K, T, chain.rate, HESTON, "call", 0.013),
                                     chain.spot, K, T, chain.rate, "call", 0.013)
        assert surface.vol(K, T) == pytest.approx(true_iv, abs=0.004)


def test_interpolated_total_variance_is_monotone_in_T(surface):
    Ts = np.linspace(surface.maturities[0], surface.maturities[-1], 60)
    for y in (-0.2, 0.0, 0.1):
        w = [float(surface.total_variance(y, t)) for t in Ts]
        assert np.all(np.diff(w) > 0)


def test_vol_extrapolates_flat_and_broadcasts(surface):
    K = surface.forward(0.02)
    assert surface.vol(K, 0.02) == pytest.approx(surface.vol(K, 0.01), rel=0.05)
    out = surface.vol(np.array([450.0, 500.0, 550.0]), 0.5)
    assert out.shape == (3,) and out[0] > out[1] > out[2]  # equity skew
    with pytest.raises(ValueError):
        surface.vol(500.0, 0.0)


def test_local_vol_is_positive_on_the_data_range(surface):
    for T in np.linspace(surface.maturities[0] + 0.01, surface.maturities[-1] - 0.01, 8):
        for k in np.linspace(-0.2, 0.15, 8):
            lv = surface.local_vol(surface.forward(T) * math.exp(k), T)
            assert lv > 0.05 and np.isfinite(lv)
    assert surface.diagnostics()["local_vol_invalid_pct"] == 0.0


def test_local_vol_equals_implied_vol_for_a_flat_surface():
    sigma = 0.2
    slices = [
        SmileSlice(f"e{i}", T, 100.0, 0.0, SviParams(sigma**2 * T, 1e-7, 0.0, 0.0, 0.1), 10, 0.0, -0.2, 0.2)
        for i, T in enumerate((0.25, 0.5, 1.0))
    ]
    surf = VolSurface.from_slices("FLAT", 100.0, 0.0, slices)
    for K, T in [(100.0, 0.4), (90.0, 0.7), (110.0, 0.3)]:
        assert surf.vol(K, T) == pytest.approx(sigma, abs=1e-4)
        assert surf.local_vol(K, T) == pytest.approx(sigma, abs=2e-3)


def test_local_vol_nan_when_surface_violates_calendar():
    """Total variance decreasing in T gives dw/dT < 0: no valid local vol, flagged as NaN."""
    p_hi = SviParams(0.04, 1e-7, 0.0, 0.0, 0.1)
    p_lo = SviParams(0.02, 1e-7, 0.0, 0.0, 0.1)
    slices = [SmileSlice("a", 0.25, 100.0, 0.0, p_hi, 10, 0, -0.2, 0.2),
              SmileSlice("b", 0.5, 100.0, 0.0, p_lo, 10, 0, -0.2, 0.2)]
    surf = VolSurface.from_slices("BAD", 100.0, 0.0, slices)
    assert np.isnan(surf.local_vol(100.0, 0.4))


def test_calendar_violation_is_detected_and_repaired(prepared):
    """Corrupt the 2nd-expiry market vols downwards so the raw fit crosses the 30d slice, then repair."""
    ivs = implied_vols_from_quotes(prepared)
    first, second = prepared.forwards.sort_values("T").expiry.iloc[:2]
    bad = ivs.copy()
    mask = bad.expiry == second
    bad.loc[mask, "iv"] *= 0.60
    bad.loc[mask, "w"] = bad.loc[mask, "iv"] ** 2 * bad.loc[mask, "T"]

    raw = fit_slices(bad, prepared.forwards, repair=False)
    assert raw[0].calendar_ok_to_next is False

    fixed = fit_slices(bad, prepared.forwards, repair=True)
    assert fixed[1].repaired
    assert fixed[0].calendar_ok_to_next is True
    assert fixed[1].butterfly_free
    k = np.linspace(-0.2, 0.15, 100)
    assert np.all(svi_total_variance(k, fixed[1].params) >= svi_total_variance(k, fixed[0].params) - 1e-8)


def test_build_needs_two_expiries(prepared):
    import dataclasses

    first = prepared.forwards.iloc[:1]
    one = dataclasses.replace(prepared, forwards=first, otm=prepared.otm[prepared.otm.expiry == first.expiry.iloc[0]])
    with pytest.raises(ValueError):
        build_surface(one)


def test_calendar_ok_helper():
    a, b = SviParams(0.02, 0.1, -0.3, 0, 0.2), SviParams(0.05, 0.1, -0.3, 0, 0.2)
    k = np.linspace(-0.5, 0.5, 50)
    assert _calendar_ok(a, b, k) and not _calendar_ok(b, a, k)
