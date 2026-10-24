"""Tests for src/pricing/lsm.py and src/pricing/pde.py."""

import pytest

from pricing.binomial import binomial_price
from pricing.black_scholes import black_scholes_price
from pricing.lsm import lsm_price
from pricing.pde import pde_price

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pde_european_matches_black_scholes(option_type) -> None:
    bs = black_scholes_price(S0, K, T, R, SIGMA, option_type, Q)
    assert pde_price(S0, K, T, R, SIGMA, option_type, Q) == pytest.approx(bs, abs=2e-3)


def test_pde_second_order_convergence() -> None:
    bs = black_scholes_price(S0, K, T, R, SIGMA, "call", Q)
    coarse = abs(pde_price(S0, K, T, R, SIGMA, "call", Q, n_space=100, n_time=100) - bs)
    fine = abs(pde_price(S0, K, T, R, SIGMA, "call", Q, n_space=400, n_time=400) - bs)
    assert fine < coarse / 4


def test_pde_american_put_matches_binomial() -> None:
    tree = binomial_price(S0, K, T, R, SIGMA, "put", Q, n_steps=2000, exercise_style="american")
    pde = pde_price(S0, K, T, R, SIGMA, "put", Q, exercise_style="american", n_space=600, n_time=600)
    assert pde == pytest.approx(tree, abs=0.01)


def test_pde_american_call_without_dividends_equals_european() -> None:
    eur = pde_price(S0, K, T, R, SIGMA, "call", 0.0, "european")
    amer = pde_price(S0, K, T, R, SIGMA, "call", 0.0, "american")
    assert amer == pytest.approx(eur, abs=1e-6)


def test_lsm_american_put_close_to_tree_and_not_above_it() -> None:
    tree = binomial_price(S0, K, T, R, SIGMA, "put", Q, n_steps=2000, exercise_style="american")
    res = lsm_price(S0, K, T, R, SIGMA, "put", Q, n_paths=100_000, n_steps=50)
    assert res.price == pytest.approx(tree, abs=0.1)
    assert res.price < tree + 3 * res.std_error  # low-biased estimator


def test_lsm_exceeds_european_put() -> None:
    eur = black_scholes_price(S0, K, T, R, SIGMA, "put", Q)
    assert lsm_price(S0, K, T, R, SIGMA, "put", Q, n_paths=100_000).price > eur
