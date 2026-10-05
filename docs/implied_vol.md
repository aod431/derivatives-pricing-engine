# Implied volatility

Implementation: [`src/pricing/implied_vol.py`](../src/pricing/implied_vol.py)
Verification notebook: [`notebooks/04_implied_volatility.ipynb`](../notebooks/04_implied_volatility.ipynb)

## 1. The inverse problem

Black-Scholes normally runs forward: given `S, K, T, r, sigma, q`, compute a price. In practice the market quotes a price, and you want to recover the `sigma` consistent with it — inverting the function. Fixing everything except `sigma`, this is a root-finding problem:

```
Price(sigma) - observed_price = 0
```

## 2. Existence and uniqueness

`Price(sigma)` is strictly increasing in `sigma` because Vega (`dPrice/dsigma`) is always positive (see [`docs/greeks.md`](greeks.md)) — a strictly increasing function is injective, so at most one `sigma` can reproduce a given price.

A solution exists only if the price lies within the **no-arbitrage bounds**:

- As `sigma -> 0+`, the price converges to the discounted forward intrinsic value (the deterministic, zero-uncertainty case).
- As `sigma -> infinity`, a call converges to `S*e^(-qT)` and a put converges to `K*e^(-rT)` — the maximum amount either option can ever be worth, regardless of volatility.

Any price outside that open interval cannot correspond to any finite, positive volatility under Black-Scholes; `implied_volatility()` checks this explicitly and raises `ValueError` rather than attempting to solve.

## 3. Brent's method vs. Newton-Raphson

`implied_volatility()` uses `scipy.optimize.brentq`, which combines bisection (guaranteed convergence, slow) with inverse quadratic interpolation (fast). It only needs a bracketing interval where the price-difference function changes sign — the default `(1e-6, 5.0)` (up to 500% vol) covers any realistic case. Newton-Raphson converges faster when it works, but needs Vega at every step and can diverge when Vega is very small (deep ITM/OTM options, very short maturities) or the starting guess is far from the root. Brent trades some speed for a correctness guarantee.

## 4. Worked example: round-trip recovery

Parameters: `S0=100, K=100, T=1, r=5%, q=2%`, call option. A price is generated at a *known* `true_sigma` via `black_scholes_price`, then fed back into `implied_volatility` to see if it recovers that same `true_sigma`:

| true_sigma | synthetic_price | recovered_sigma | abs_error |
|---|---|---|---|
| 0.05 | 3.711144 | 0.05 | 2.41e-15 |
| 0.10 | 5.471349 | 0.10 | 1.34e-11 |
| 0.15 | 7.336873 | 0.15 | 6.72e-13 |
| 0.20 | 9.227006 | 0.20 | 2.50e-15 |
| 0.30 | 13.020281 | 0.30 | 6.33e-15 |
| 0.50 | 20.546473 | 0.50 | 2.11e-15 |
| 0.80 | 31.487055 | 0.80 | 3.33e-16 |
| 1.20 | 45.061915 | 1.20 | 3.27e-13 |

All errors are at or near `float64` machine precision (~1e-15 to 1e-11) — the solver recovers the input volatility essentially exactly, across a wide range from 5% to 120%, confirming `implied_volatility` is a correct numerical inverse of `black_scholes_price`.

The same round-trip was repeated across a range of spots (deep ITM to deep OTM, both call and put) and with a synthetic, strike-dependent volatility ("smile") shape — in every case the recovered curve matched the input to within `1e-6`.

## 5. Explicit rejection of an impossible price

With the same parameters, a call priced at `150.0` is rejected:

```
Price 150.0 is outside the no-arbitrage bounds (2.896925, 98.019867) for
this option: no finite positive volatility can reproduce it.
```

- **Upper bound (98.019867)**: `S*e^(-qT) = 100*e^(-0.02)`. A call can never be worth more than the (dividend-discounted) underlying itself — buying it outright would always be cheaper, which would be a trivial arbitrage. `150 > 98.02`, so it is rejected.
- **Lower bound (2.896925)**: the discounted forward intrinsic value, `e^(-rT)*max(F-K, 0)` with `F = S*e^((r-q)T) = 103.045`. Since `r > q`, the forward sits above the strike even though spot is at-the-money, so the zero-volatility floor is strictly positive rather than 0 (the same quantity seen as `C - P` in the put-call parity worked example in [`docs/black_scholes.md`](black_scholes.md)).
