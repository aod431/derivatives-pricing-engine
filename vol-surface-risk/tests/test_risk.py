"""Risk: pricing consistency, Greeks vs finite differences, bucketing, VaR approximation."""

import numpy as np
import pytest

from pricing.black_scholes import black_scholes_price
from pricing.greeks import analytical_greeks
from volrisk.risk import (
    PortfolioPricer, Position, bs_vec, pnl_delta_gamma_vega, pnl_full_revaluation, risk_report,
    simulate_risk_factors, var_es,
)

POSITIONS = [
    Position("call", 520.0, 0.25, -100), Position("put", 480.0, 0.25, -100),
    Position("call", 500.0, 0.50, 50), Position("put", 450.0, 1.00, 80),
]


@pytest.fixture(scope="module")
def pricer(surface):
    return PortfolioPricer(surface, POSITIONS)


def test_vectorised_bs_matches_scalar_library():
    for opt in ("call", "put"):
        p, d, g, v, th = bs_vec(100.0, 105.0, 0.7, 0.03, 0.25, 0.01, opt == "call")
        assert p == pytest.approx(black_scholes_price(100, 105, 0.7, 0.03, 0.25, opt, 0.01))
        ref = analytical_greeks(100, 105, 0.7, 0.03, 0.25, opt, 0.01)
        assert (d, g, v, th) == pytest.approx((ref.delta, ref.gamma, ref.vega, ref.theta))


def test_position_validation():
    with pytest.raises(ValueError):
        Position("straddle", 100, 1, 1)
    with pytest.raises(ValueError):
        Position("call", -1, 1, 1)
    with pytest.raises(ValueError):
        Position("call", 100, 0, 1)
    with pytest.raises(ValueError):
        PortfolioPricer(None, [])


def test_option_priced_at_surface_vol(surface):
    pr = PortfolioPricer(surface, [Position("call", 500.0, 0.5, 1)])
    expected = black_scholes_price(surface.spot, 500.0, 0.5, surface.rate, surface.vol(500.0, 0.5), "call",
                                   surface.carry(0.5))
    assert pr.base_value == pytest.approx(expected)


def test_portfolio_greeks_match_finite_differences(pricer):
    g, S = pricer.greeks(), pricer.S
    h = 1e-3
    up, dn, mid = pricer.value(spot_pct=h), pricer.value(spot_pct=-h), pricer.base_value
    assert g["delta"] == pytest.approx((up - dn) / (2 * h * S), rel=1e-4)
    assert g["gamma"] == pytest.approx((up - 2 * mid + dn) / (h * S) ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx(pricer.value(vol_shift=0.005) - pricer.value(vol_shift=-0.005), rel=1e-3)
    assert g["theta"] == pytest.approx(pricer.value(dt_years=1 / 365) - mid, rel=5e-3)


def test_bucketed_vega_sums_to_parallel_vega(pricer):
    buckets = pricer.bucketed_vega()
    assert sum(buckets.values()) == pytest.approx(pricer.greeks()["vega"], rel=1e-3)
    assert set(buckets) == {s.expiry for s in pricer.surface.slices}


def test_bucketed_vega_lands_on_the_right_expiry(surface):
    T = float(surface.maturities[3])
    pr = PortfolioPricer(surface, [Position("call", 500.0, T, 1)])
    buckets = pr.bucketed_vega()
    target = surface.slices[3].expiry
    assert buckets[target] == pytest.approx(pr.greeks()["vega"], rel=1e-3)
    assert all(abs(v) < 1e-9 for e, v in buckets.items() if e != target)


def test_scenario_grid_properties(pricer):
    grid = pricer.scenario_grid([-0.05, 0.0, 0.05], [-0.02, 0.0, 0.02])
    assert grid.loc[0.0, 0.0] == pytest.approx(0.0, abs=1e-9)
    assert grid.shape == (3, 3)
    # net long vega: raising vol with spot unchanged must help
    assert grid.loc[0.0, 0.02] > 0 > grid.loc[0.0, -0.02]
    # skew tilt is a real scenario: P&L differs from the no-shock case
    assert pricer.scenario_grid([0.0], [0.0], vol_skew=0.1).iloc[0, 0] != 0.0


def test_dgv_matches_full_revaluation_on_small_shocks(pricer):
    """With |spot move| < 0.5% and |dvol| < 0.2 pt the Taylor approximation is within 5% of the P&L scale."""
    rng = np.random.default_rng(0)
    ret, dvol = rng.uniform(-0.005, 0.005, 400), rng.uniform(-0.002, 0.002, 400)
    full = pnl_full_revaluation(pricer, ret, dvol)
    approx = pnl_delta_gamma_vega(pricer, ret, dvol)
    assert np.max(np.abs(full - approx)) < 0.05 * np.max(np.abs(full))


def test_var_dgv_close_to_full_revaluation_for_a_directional_book(surface):
    """Delta-dominated book: second-order Taylor is an excellent VaR/ES approximation."""
    book = PortfolioPricer(surface, [Position("call", 500.0, 0.5, 100), Position("put", 480.0, 0.25, -50),
                                     Position("call", 540.0, 1.0, -40)])
    rep = risk_report(book, spot_vol=0.18, vol_sd=0.01, n_sims=20_000)
    f, a = rep["full_revaluation"], rep["delta_gamma_vega"]
    assert a["var"] == pytest.approx(f["var"], rel=0.02)
    assert a["es"] == pytest.approx(f["es"], rel=0.02)
    assert f["es"] >= f["var"] > 0


def test_var_dgv_is_materially_worse_for_a_short_gamma_book(pricer):
    """Short strangle + vol shocks: spot-vol cross terms (vanna) and higher-order gamma are ignored."""
    rep = risk_report(pricer, n_sims=20_000)
    assert abs(rep["approx_error_pct_var"]) > 5.0


def test_var_es_basics():
    pnl = np.arange(-100.0, 100.0)  # uniform P&L
    var, es = var_es(pnl, 0.95)
    assert var == pytest.approx(90.0, abs=1.5) and es > var
    with pytest.raises(ValueError):
        var_es(pnl, 0.3)


def test_simulated_factors_have_requested_moments():
    ret, dvol = simulate_risk_factors(200_000, spot_vol=0.20, vol_sd=0.01, corr=-0.7, tail_df=8, seed=1)
    assert ret.std() == pytest.approx(0.20 / np.sqrt(252), rel=0.05)
    assert dvol.std() == pytest.approx(0.01, rel=0.05)
    assert np.corrcoef(ret, dvol)[0, 1] == pytest.approx(-0.7, abs=0.03)
    with pytest.raises(ValueError):
        simulate_risk_factors(10, 0.2, 0.01, tail_df=2.0)
