"""Tests for src/pricing/mc_greeks.py, quasi_mc.py and hedging.py."""

import numpy as np
import pytest

from pricing.black_scholes import black_scholes_price
from pricing.greeks import analytical_greeks
from pricing.hedging import delta_hedge_pnl
from pricing.mc_greeks import mc_delta, mc_vega
from pricing.monte_carlo import mc_price
from pricing.quasi_mc import qmc_price

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


@pytest.mark.parametrize("method", ["pathwise", "likelihood_ratio", "bump"])
def test_call_delta_estimators_match_analytical(method) -> None:
    exact = analytical_greeks(S0, K, T, R, SIGMA, "call", Q).delta
    est = mc_delta(S0, K, T, R, SIGMA, "call", Q, method=method, n_paths=400_000)
    assert abs(est.value - exact) < 4 * est.std_error + 1e-3


def test_pathwise_vega_matches_analytical() -> None:
    exact = analytical_greeks(S0, K, T, R, SIGMA, "call", Q).vega
    est = mc_vega(S0, K, T, R, SIGMA, "call", Q, n_paths=400_000)
    assert abs(est.value - exact) < 4 * est.std_error


def test_digital_delta_via_likelihood_ratio_and_pathwise_rejected() -> None:
    from scipy.stats import norm

    d2 = (np.log(S0 / K) + (R - Q - 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    exact = np.exp(-R * T) * norm.pdf(d2) / (S0 * SIGMA * np.sqrt(T))
    est = mc_delta(S0, K, T, R, SIGMA, "digital_call", Q, method="likelihood_ratio", n_paths=800_000)
    assert abs(est.value - exact) < 4 * est.std_error
    with pytest.raises(ValueError):
        mc_delta(S0, K, T, R, SIGMA, "digital_call", Q, method="pathwise")


def test_common_random_numbers_cut_bump_variance() -> None:
    crn = mc_delta(S0, K, T, R, SIGMA, "call", Q, method="bump", common_random_numbers=True)
    indep = mc_delta(S0, K, T, R, SIGMA, "call", Q, method="bump", common_random_numbers=False)
    assert indep.std_error > 10 * crn.std_error


def test_qmc_beats_plain_mc_at_equal_budget() -> None:
    bs = black_scholes_price(S0, K, T, R, SIGMA, "call", Q)
    q = qmc_price(S0, K, T, R, SIGMA, "call", Q, log2_paths=12, n_replicates=16)
    plain = mc_price(S0, K, T, R, SIGMA, "call", Q, n_paths=q.n_paths)
    assert abs(q.price - bs) < abs(plain.price - bs) or q.std_error < plain.std_error / 5
    assert q.std_error < plain.std_error / 5


def test_hedge_pnl_std_shrinks_with_rebalancing_frequency() -> None:
    few = delta_hedge_pnl(S0, K, T, R, SIGMA, SIGMA, 12, n_paths=20_000)
    many = delta_hedge_pnl(S0, K, T, R, SIGMA, SIGMA, 192, n_paths=20_000)
    assert many.std() < few.std() / 3  # ~ sqrt(16) = 4x lower


def test_hedge_pnl_sign_follows_implied_minus_realised_vol() -> None:
    sold_rich = delta_hedge_pnl(S0, K, T, R, 0.30, 0.20, 100, n_paths=20_000)
    sold_cheap = delta_hedge_pnl(S0, K, T, R, 0.20, 0.30, 100, n_paths=20_000)
    assert sold_rich.mean() > 0 > sold_cheap.mean()


def test_hedge_pnl_mean_near_zero_when_vols_match() -> None:
    pnl = delta_hedge_pnl(S0, K, T, R, SIGMA, SIGMA, 200, n_paths=20_000)
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(pnl.size)
