# Black-Scholes model

Implementation: [`src/pricing/black_scholes.py`](../src/pricing/black_scholes.py)
Verification notebook: [`notebooks/01_black_scholes.ipynb`](../notebooks/01_black_scholes.ipynb)

## 1. Price model: geometric Brownian motion (GBM)

The underlying price $S_t$ is assumed to follow

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

The instantaneous return has a deterministic drift term ($\mu\,dt$) plus a random term proportional to the price itself ($\sigma\,dW_t$). This keeps prices strictly positive and makes *percentage* returns (not absolute price changes) the quantity with constant volatility. Solving the SDE gives

$$S_T = S_0 \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T\right]$$

so $S_T$ is **log-normally** distributed: $\ln S_T$ is normal.

## 2. Risk-neutral valuation

The key Black-Scholes-Merton insight: a self-financing portfolio of the underlying and a risk-free bond can replicate the option's payoff at every instant (delta hedging). By no-arbitrage, this riskless portfolio must earn exactly the risk-free rate $r$. This means **the real-world drift $\mu$ never enters the price** — the option value does not depend on whether investors expect the stock to go up or down, only on its volatility.

Formally, there exists a risk-neutral measure $\mathbb{Q}$ under which the discounted price is a martingale, and the drift of $S_t$ becomes $r - q$ (risk-free rate minus the continuous dividend yield $q$):

$$\text{Price}_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_T)]$$

## 3. Closed-form formula

Solving the expectation above for European calls and puts:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

with

$$d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$N(d_2)$ is the risk-neutral probability that the call finishes in-the-money ($S_T > K$). $N(d_1)$ is that same concept weighted by the underlying's value in that scenario — it is also the call's Delta (see [`greeks.py`](../src/pricing/greeks.py), upcoming stage).

## 4. Assumptions

- $r$, $\sigma$, $q$ are constant and known.
- No arbitrage; frictionless markets (no transaction costs or taxes).
- Continuous trading, infinitely divisible assets.
- $S_T$ is log-normal: no jumps, constant volatility (contradicted empirically by the observed volatility smile).
- European exercise only (at $T$, not before).
- Dividends are paid as a continuous yield $q$ (an approximation; discrete dividends require an adjustment).

## 5. Position payoffs and the synthetic forward

| Position | Premium today | Payoff at $T$ | Max loss | Max gain |
|---|---|---|---|---|
| Long call | Pay | $\max(S_T-K,0)$ | Premium | Unlimited |
| Short call | Receive | $-\max(S_T-K,0)$ | Unlimited | Premium |
| Long put | Pay | $\max(K-S_T,0)$ | Premium | Large (capped at $K$) |
| Short put | Receive | $-\max(K-S_T,0)$ | Large (capped at $K$) | Premium |

Combining a long call with a short put (same $K$, same $T$) reproduces exactly the payoff of a forward contract bought at $K$:

$$\underbrace{\max(S_T-K,0)}_{\text{long call}} + \underbrace{-\max(K-S_T,0)}_{\text{short put}} = S_T - K \quad \text{in every scenario}$$

This is the mechanical reason put-call parity holds: $C - P$ must equal the present value of that synthetic forward, $S_0 e^{-qT} - K e^{-rT}$ — not a coincidence, a replication argument. See the payoff diagrams in [`notebooks/01_black_scholes.ipynb`](../notebooks/01_black_scholes.ipynb) (section 3), which also verifies the synthetic-forward identity numerically.

The fair (zero-cost) forward price is $F = S_0 e^{(r-q)T}$ — the delivery price that makes entering the forward free today. If the option's strike $K$ differs from $F$, the synthetic forward has nonzero present value $e^{-rT}(F-K)$, which is exactly why $C \neq P$ whenever $r \neq q$, even at $S_0 = K$ (see the worked example below, where $r=5\%>q=2\%$ pushes $F\approx103.05 > K=100$, making the call more valuable than the put).

## 6. Worked example

Parameters used in [`notebooks/01_black_scholes.ipynb`](../notebooks/01_black_scholes.ipynb):

| Parameter | Value |
|---|---|
| Spot $S_0$ | 100.0 |
| Strike $K$ | 100.0 |
| Maturity $T$ | 1.0 year |
| Risk-free rate $r$ | 5% |
| Volatility $\sigma$ | 20% |
| Dividend yield $q$ | 2% |

Results:

| Quantity | Value |
|---|---|
| Call price | 9.227006 |
| Put price | 6.330081 |
| $C - P$ | 2.896925 |
| $S_0 e^{-qT} - K e^{-rT}$ | 2.896925 |
| Absolute difference | 0.00e+00 |

The put-call parity relation $C - P = S_0 e^{-qT} - K e^{-rT}$ holds exactly, which is a basic sanity check that the implementation is internally consistent (an at-the-money call is more expensive than the equivalent put here because the risk-free rate exceeds the dividend yield, $r > q$, which makes holding the underlying synthetically via calls preferable to the put side).
