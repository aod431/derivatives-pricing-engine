"""Tests for src/pricing/barrier.py: parity, limits and agreement with Monte Carlo."""

import pytest

from pricing.barrier import barrier_mc_price, barrier_price
from pricing.black_scholes import black_scholes_price

S0, T, R, SIGMA, Q = 100.0, 1.0, 0.05, 0.2, 0.0
CASES = [
    ("call", "down-and-out", 100.0, 90.0),
    ("call", "up-and-out", 100.0, 130.0),
    ("put", "down-and-out", 100.0, 85.0),
    ("put", "up-and-out", 100.0, 120.0),
    ("call", "down-and-in", 100.0, 90.0),
    ("call", "up-and-in", 100.0, 130.0),
    ("put", "down-and-in", 100.0, 85.0),
    ("put", "up-and-in", 100.0, 120.0),
]


@pytest.mark.parametrize("option_type,barrier_type,K,H", CASES)
def test_closed_form_matches_bridge_monte_carlo(option_type, barrier_type, K, H) -> None:
    exact = barrier_price(S0, K, H, T, R, SIGMA, option_type, barrier_type, Q)
    mc = barrier_mc_price(S0, K, H, T, R, SIGMA, option_type, barrier_type, Q, n_paths=400_000, n_steps=100)
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.01


@pytest.mark.parametrize("option_type,K,H_dn,H_up", [("call", 100.0, 90.0, 130.0), ("put", 100.0, 85.0, 120.0)])
def test_in_out_parity(option_type, K, H_dn, H_up) -> None:
    vanilla = black_scholes_price(S0, K, T, R, SIGMA, option_type, Q)
    for kind, H in (("down", H_dn), ("up", H_up)):
        total = barrier_price(S0, K, H, T, R, SIGMA, option_type, f"{kind}-and-in", Q) + barrier_price(
            S0, K, H, T, R, SIGMA, option_type, f"{kind}-and-out", Q
        )
        assert total == pytest.approx(vanilla, abs=1e-10)


def test_knock_out_is_cheaper_than_vanilla_and_far_barrier_converges() -> None:
    vanilla = black_scholes_price(S0, 100.0, T, R, SIGMA, "call", Q)
    near = barrier_price(S0, 100.0, 95.0, T, R, SIGMA, "call", "down-and-out", Q)
    far = barrier_price(S0, 100.0, 1.0, T, R, SIGMA, "call", "down-and-out", Q)
    assert near < vanilla
    assert far == pytest.approx(vanilla, abs=1e-6)


def test_bridge_correction_removes_discrete_monitoring_bias() -> None:
    """Without the bridge, MC overprices a knock-out; with it, it matches the closed form."""
    exact = barrier_price(S0, 100.0, 90.0, T, R, SIGMA, "call", "down-and-out", Q)
    naive = barrier_mc_price(S0, 100.0, 90.0, T, R, SIGMA, "call", "down-and-out", Q,
                             n_paths=400_000, n_steps=12, brownian_bridge=False)
    fixed = barrier_mc_price(S0, 100.0, 90.0, T, R, SIGMA, "call", "down-and-out", Q,
                             n_paths=400_000, n_steps=12, brownian_bridge=True)
    assert naive.price - exact > 5 * naive.std_error
    assert abs(fixed.price - exact) < 4 * fixed.std_error + 0.01


def test_barrier_side_validation() -> None:
    with pytest.raises(ValueError):
        barrier_price(S0, 100.0, 110.0, T, R, SIGMA, "call", "down-and-out")
    with pytest.raises(ValueError):
        barrier_price(S0, 100.0, 90.0, T, R, SIGMA, "call", "sideways")
