# Greeks

Implementation: [`src/pricing/greeks.py`](../src/pricing/greeks.py)
Verification notebook: [`notebooks/03_greeks.ipynb`](../notebooks/03_greeks.ipynb)

## 1. What Greeks measure

Greeks are the partial derivatives of the option price with respect to each input: how much the price moves if the spot, volatility, time, or rate moves a little. They are the basis of risk management on a derivatives desk — a trader hedges the Greeks, not just the price.

- **Delta** (`dPrice/dS`): sensitivity to the spot. Also the hedge ratio (units of underlying needed to delta-hedge the option).
- **Gamma** (`d^2Price/dS^2`): sensitivity of delta to the spot. High gamma means the hedge needs to be rebalanced frequently.
- **Vega** (`dPrice/dsigma`): sensitivity to volatility.
- **Theta** (`-dPrice/dT`): time decay — value lost per year as time passes, all else equal.
- **Rho** (`dPrice/dr`): sensitivity to the risk-free rate.

## 2. Closed-form formulas

With `d1`, `d2` as in [`docs/black_scholes.md`](black_scholes.md) and `phi` the standard normal density:

```
Gamma       = e^(-qT) * phi(d1) / (S * sigma * sqrt(T))          (same for call and put)
Vega        = S * e^(-qT) * phi(d1) * sqrt(T)                    (same for call and put)

Delta_call  =  e^(-qT) * N(d1)
Delta_put   =  e^(-qT) * (N(d1) - 1)

Theta_call  = -S*e^(-qT)*phi(d1)*sigma / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)  + q*S*e^(-qT)*N(d1)
Theta_put   = -S*e^(-qT)*phi(d1)*sigma / (2*sqrt(T)) + r*K*e^(-rT)*N(-d2) - q*S*e^(-qT)*N(-d1)

Rho_call    =  K*T*e^(-rT)*N(d2)
Rho_put     = -K*T*e^(-rT)*N(-d2)
```

These come from differentiating the Black-Scholes call/put formula directly (`N(d1)`, `N(d2)` depend on `S` and `sigma` through `d1`, `d2`).

## 3. What put-call parity tells you about them for free

Differentiating the parity identity `C - P = S*e^(-qT) - K*e^(-rT)` with respect to `S` repeatedly gives relationships between the call and put Greeks, without re-deriving each one from scratch:

- First derivative: `Delta_C - Delta_P = e^(-qT)` — a **constant** offset, independent of `S` and `sigma`.
- Second derivative: `Gamma_C - Gamma_P = 0` — Gamma is identical for calls and puts.
- Derivative w.r.t. `sigma`: `Vega_C - Vega_P = 0` — Vega is identical for calls and puts.
- Derivative w.r.t. `T`: `Theta_C - Theta_P = -q*S*e^(-qT) + r*K*e^(-rT)` — depends on `S`, `K`, `r`, `q`, `T`, so theta is **not** simply offset between call and put.

Parity only proves the *relationship*; the actual formulas above still come from differentiating the full pricing formula.

## 4. Finite-difference validation, and why gamma is noisier

Any pricer without a closed form (Monte Carlo, binomial, exotics) can only get Greeks via finite differences: bump an input by `h` and measure the price change.

```
Delta ~ [Price(S+h) - Price(S-h)] / (2h)
Gamma ~ [Price(S+h) - 2*Price(S) + Price(S-h)] / h^2
```

Delta divides by `2h`; Gamma divides by `h^2`. Floating-point rounding error in the subtracted prices (irreducible, ~1e-15 per price) gets amplified far more by `1/h^2` than by `1/h`. There is a classic trade-off: `h` too large biases the estimate (truncation error, the price curve is not exactly quadratic over a wide step); `h` too small lets rounding error dominate (catastrophic cancellation). `tests/test_greeks.py` uses `h=0.01` for delta but `h=0.5` for gamma — a smaller `h` for gamma would make the finite-difference estimate *worse*, not better.

## 5. Worked example

Parameters: `S0=100, K=100, T=1, r=5%, sigma=20%, q=2%` (same as previous notebooks).

```
Call Greeks: delta=0.586851, gamma=0.018951, vega=37.901158, theta=-5.089319, rho=49.458109
Put Greeks:  delta=-0.393348, gamma=0.018951, vega=37.901158, theta=-2.293569, rho=-45.664833
```

Checks against the theory above:
- `Delta_call - Delta_put = 0.980199 = e^(-qT) = e^(-0.02)` — matches exactly, as predicted.
- `gamma` and `vega` are identical between call and put, as predicted.
- `rho` is positive for the call and negative for the put: a higher `r` raises the forward (`F = S*e^((r-q)T)`), which helps a call (bets on a higher `S_T`) and hurts a put, and discounts the strike leg more, which also favors the call.

**Gamma peaks to the left of the strike.** Differentiating `Gamma(S)` and solving for its maximum gives:

```
S_peak = K * exp[-(r - q + 1.5*sigma^2) * T]
```

With the parameters above: `100 * exp(-(0.05 - 0.02 + 1.5*0.04)) = 100 * exp(-0.09) ≈ 91.4` — matching the peak observed in the notebook's Gamma-vs-spot plot. Since `1.5*sigma^2` is always positive, `S_peak < K` for almost any realistic combination of `r`, `q`, `sigma`; the peak would only move to the right of `K` if `q` were deeply larger than `r + 1.5*sigma^2`, an unusual case (very high dividend yield, low rates and low vol).
