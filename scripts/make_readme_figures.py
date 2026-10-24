"""Generate the figures and the validation table used in README.md and docs/.

Run from the repository root:  python scripts/make_readme_figures.py
Figures go to docs/img/. The validation table is printed to stdout.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricing.binomial import binomial_price
from pricing.black_scholes import black_scholes_price
from pricing.greeks import analytical_greeks, finite_difference_derivative
from pricing.hedging import delta_hedge_pnl
from pricing.heston import HestonParams, heston_mc_price, heston_price
from pricing.implied_vol import implied_volatility
from pricing.lsm import lsm_price
from pricing.monte_carlo import mc_price
from pricing.pde import pde_price
from pricing.quasi_mc import qmc_price

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
BS = black_scholes_price(S0, K, T, R, SIGMA, "call", Q)


def mc_convergence() -> None:
    ns = np.unique(np.logspace(2, 6, 13).astype(int))
    se, err = [], []
    for n in ns:
        res = mc_price(S0, K, T, R, SIGMA, "call", Q, n_paths=int(n), seed=7)
        se.append(res.std_error)
        err.append(abs(res.price - BS))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ns, se, "o-", label="standard error")
    ax.loglog(ns, err, "s", alpha=0.6, label="|MC price - Black-Scholes|")
    ax.loglog(ns, se[0] * np.sqrt(ns[0]) / np.sqrt(ns), "k--", linewidth=1, label=r"$1/\sqrt{N}$")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Error (price units)")
    ax.set_title("Monte Carlo converges to Black-Scholes at rate 1/sqrt(N)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mc_convergence.png", dpi=130)
    plt.close(fig)


def heston_smiles() -> None:
    strikes = np.linspace(70, 130, 25)
    fwd = S0 * np.exp((R - Q) * T)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rho in (-0.7, 0.0, 0.5):
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=rho)
        vols = []
        for k in strikes:
            opt = "put" if k < fwd else "call"
            vols.append(implied_volatility(heston_price(S0, k, T, R, p, opt, Q), S0, k, T, R, opt, Q))
        ax.plot(strikes, 100 * np.array(vols), label=rf"$\rho = {rho}$")
    ax.axhline(100 * SIGMA, color="grey", linestyle=":", label="Black-Scholes (flat)")
    ax.set_xlabel("Strike (spot = 100)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Heston: the correlation rho sets the skew")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "heston_smile.png", dpi=130)
    plt.close(fig)


def hedging_hist() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in (12, 52, 252):
        pnl = delta_hedge_pnl(S0, K, T, R, SIGMA, SIGMA, n, n_paths=20_000)
        ax.hist(pnl, bins=80, range=(-8, 8), alpha=0.5, label=f"{n} rebalances (std {pnl.std():.2f})")
    ax.set_xlabel("Hedging P&L at maturity")
    ax.set_ylabel("Paths")
    ax.set_title("Delta hedging a short call: P&L spread shrinks like 1/sqrt(n)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hedging_pnl.png", dpi=130)
    plt.close(fig)


def validation_table() -> None:
    rows = [("Black-Scholes (reference)", BS, None, "closed form")]
    rows.append(("Monte Carlo, 1,000,000 paths", (r := mc_price(S0, K, T, R, SIGMA, "call", Q, n_paths=1_000_000, antithetic=True, seed=42)).price,
                 None, f"SE {r.std_error:.4f}"))
    rows.append(("Sobol QMC, 16 x 2^14 points", (r := qmc_price(S0, K, T, R, SIGMA, "call", Q, log2_paths=14, n_replicates=16)).price, None, f"SE {r.std_error:.5f}"))
    rows.append(("Binomial tree, 1000 steps", binomial_price(S0, K, T, R, SIGMA, "call", Q, n_steps=1000), None, ""))
    rows.append(("Crank-Nicolson PDE, 400 x 400", pde_price(S0, K, T, R, SIGMA, "call", Q), None, ""))
    print("European call, S=K=100, T=1, r=5%, sigma=20%, q=2%")
    print("| Method | Price | Abs. error vs BS | Note |\n|---|---|---|---|")
    for name, price, _, note in rows:
        print(f"| {name} | {price:.4f} | {abs(price - BS):.4f} | {note} |")

    tree = binomial_price(S0, K, T, R, SIGMA, "put", Q, n_steps=2000, exercise_style="american")
    pde = pde_price(S0, K, T, R, SIGMA, "put", Q, exercise_style="american", n_space=600, n_time=600)
    lsm = lsm_price(S0, K, T, R, SIGMA, "put", Q, n_paths=100_000)
    print(f"\nAmerican put: tree {tree:.4f} | PDE {pde:.4f} | LSM {lsm.price:.4f} (SE {lsm.std_error:.4f})")

    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    exact = heston_price(S0, K, T, R, p, "call", Q)
    mc = heston_mc_price(S0, K, T, R, p, "call", Q, n_paths=200_000, n_steps=200)
    print(f"Heston call: Fourier {exact:.4f} | MC {mc.price:.4f} (SE {mc.std_error:.4f})")

    g = analytical_greeks(S0, K, T, R, SIGMA, "call", Q)
    kw = dict(S=S0, K=K, T=T, r=R, sigma=SIGMA, option_type="call", q=Q)
    fd = finite_difference_derivative(black_scholes_price, kw, "S", 0.01)
    print(f"Delta: analytical {g.delta:.6f} | finite difference {fd:.6f}")


if __name__ == "__main__":
    mc_convergence()
    heston_smiles()
    hedging_hist()
    validation_table()
