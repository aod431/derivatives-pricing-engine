# Exotics: Asian options

Implementation: [`src/pricing/exotics.py`](../src/pricing/exotics.py)
Verification notebook: [`notebooks/06_exotics.ipynb`](../notebooks/06_exotics.ipynb)

## 1. What an Asian option is, and why it needs Monte Carlo

An Asian option's payoff depends on the **average** underlying price over the option's life, not just the terminal price `S_T`:

```
Call payoff = max(Average - K, 0)
```

They are common in commodity and FX markets: averaging makes the payoff harder to manipulate near expiry (a concern in thinner markets) and is naturally cheaper than a vanilla option with the same strike, since averaging reduces the effective volatility of the payoff.

The **arithmetic** average (`(1/n) * sum(S_ti)`, the ordinary mean) has **no closed-form price** under Black-Scholes: a sum of correlated lognormal variables is not itself lognormal, so the usual machinery (`N(d1)`, `N(d2)`) does not apply. It must be priced by Monte Carlo.

## 2. The geometric Asian option: a closed form that doubles as a control variate

The **geometric** average (`(prod(S_ti))^(1/n)`) *is* lognormal: the log of a product is a sum of logs, and the logs of GBM prices are normal, so a sum of normals is normal — the log of the geometric average is normal, hence the average itself is lognormal. That means a Black-Scholes-style closed form exists, with an adjusted effective volatility and drift.

For `n` equally-spaced fixing dates over `(0, T]`, the derivation (matching the discrete monitoring scheme used by the Monte Carlo simulation, not the continuous-monitoring approximation) gives:

```
t_bar       = T * (n+1) / (2n)
drift       = (r - q - 0.5*sigma^2) * t_bar
var_total   = sigma^2 * T * (n+1)*(2n+1) / (6*n^2)
E[G]        = S * exp(drift + 0.5*var_total)

d1 = (ln(S/K) + drift + var_total) / sqrt(var_total)
d2 = d1 - sqrt(var_total)

Call = e^(-rT) * (E[G]*N(d1) - K*N(d2))
Put  = e^(-rT) * (K*N(-d2) - E[G]*N(-d1))
```

As `n -> infinity`, `var_total/T -> sigma^2/3`, recovering the classic continuous-monitoring Kemna-Vorst result (effective volatility `sigma/sqrt(3)`) — this limit matching a known, independently published formula is itself a check that the discrete derivation above is correct.

## 3. Control variate: pricing the arithmetic option using the geometric one

Simulate the **same paths** and compute both averages from them. Because both are built from identical random draws, they are highly correlated path by path (the geometric average is a "smoothed" version of the arithmetic one). This lets us correct the arithmetic Monte Carlo estimate using the geometric estimator's known error:

```
Price_corrected = Price_arithmetic_MC + (Price_geometric_closed_form - Price_geometric_MC)
```

This is unbiased — `E[Price_geometric_MC] = Price_geometric_closed_form` exactly, so the correction term has mean zero — but its variance is `Var(arithmetic_payoff - geometric_payoff)` instead of `Var(arithmetic_payoff)` alone. Since the two payoffs move together almost in lockstep, their difference is far more stable than either one individually, which is what drives the variance reduction (the same idea as the antithetic variates of [`docs/monte_carlo.md`](monte_carlo.md), but far more effective here because the correlation between the two averages is much higher than the correlation exploited by antithetic sampling).

## 4. Known, provable orderings (useful as sanity checks independent of any simulation noise)

- **Geometric <= arithmetic, pathwise** (AM-GM inequality: the geometric mean of a set of positive numbers never exceeds their arithmetic mean). Since `max(x, 0)` is monotonic, this carries through directly to a **call**: `geometric_asian_price <= arithmetic_asian_price`, always — not just in expectation, but on every single simulated path.
- For a **put**, the inequality direction flips: since `G <= A` implies `K - G >= K - A`, the geometric Asian put is worth **at least as much** as the arithmetic Asian put.
- **Both Asian variants are cheaper than the vanilla option** with the same strike and maturity, since averaging strictly reduces the effective volatility seen by the payoff relative to using the terminal price alone.

## 5. Worked example

Pending: to be filled in with the actual output of [`notebooks/06_exotics.ipynb`](../notebooks/06_exotics.ipynb) once it has been executed, following this project's rule of only recording numbers that have actually been run.
